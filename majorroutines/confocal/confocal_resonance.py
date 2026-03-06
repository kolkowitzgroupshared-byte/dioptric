# -*- coding: utf-8 -*-
"""
Confocal ESR using unified confocal_base_routine + SWAB_82 pulse-gen + SWAB_20 tagger.

Requires ONE small backward-compatible patch in confocal_base_routine.main():
  step_fn can optionally return (seq_file, seq_args_string, step_freq_ghz)
and base_routine will sg.set_freq(step_freq_ghz) before running that step.

Sequence contract:
- esr_seq.py should produce EXACTLY 2 APD gates per repetition:
    gate0 = signal readout
    gate1 = reference readout
- seq_args should at least include base_args (same idea as rabi)

@author: Saroj Chand
"""
import os
import traceback
import numpy as np

import majorroutines.confocal.confocal_base_routine as base
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import NormMode, VirtualLaserKey



###plotting helper
def _sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(x), axis=axis)
    return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))

def _moving_average(y, w=5):
    y = np.asarray(y, dtype=float)
    if w is None or w <= 1:
        return y
    w = int(w)
    if w >= len(y):
        return np.full_like(y, np.nanmean(y))
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

def _edges_baseline(y, frac=0.15):
    """Baseline estimate from edges (robust for single dip scan)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    k = max(1, int(round(frac * n)))
    edges = np.concatenate([y[:k], y[-k:]])
    return float(np.nanmedian(edges))

def _initial_resonance_from_contrast(freqs_ghz, contrast, smooth_w=5):
    """contrast should be positive near resonance (e.g., 1 - sig/ref)."""
    c_s = _moving_average(contrast, smooth_w)
    idx = int(np.nanargmax(c_s))
    return float(freqs_ghz[idx]), idx

def _rough_fwhm(freqs_ghz, contrast, idx_peak, frac=0.5):
    """
    Rough FWHM on a positive peak trace (contrast). Uses half-max crossing.
    Returns FWHM in MHz (rough).
    """
    f = np.asarray(freqs_ghz, dtype=float)
    c = np.asarray(contrast, dtype=float)
    peak = c[idx_peak]
    if not np.isfinite(peak) or peak <= 0:
        return np.nan

    half = frac * peak

    # search left
    iL = idx_peak
    while iL > 0 and np.isfinite(c[iL]) and c[iL] > half:
        iL -= 1
    # search right
    iR = idx_peak
    while iR < len(c) - 1 and np.isfinite(c[iR]) and c[iR] > half:
        iR += 1

    if iL == idx_peak or iR == idx_peak:
        return np.nan

    fwhm_ghz = f[iR] - f[iL]
    return float(fwhm_ghz * 1e3)  # GHz -> MHz

def analyze_esr(
    raw: dict,
    *,
    smooth_w: int = 5,
    baseline_edge_frac: float = 0.15,
    do_plot: bool = True,
    save_path: str | None = None,
    title: str | None = None,
):
    """
    Analyze ESR/ODMR sweep from raw dict produced by confocal_esr.main().

    Expects:
      raw["counts"] shape (2, num_runs, num_steps) or raw["sig_counts"]/raw["ref_counts"]
      raw["freqs_ghz"] list length num_steps

    Returns dict with:
      f0_ghz, f0_mhz_detuning, fwhm_mhz, contrast_peak, baseline_norm, fit_params (if fit),
      traces (mean/sem), etc.
    """

    freqs_ghz = np.asarray(raw.get("freqs_ghz", None), dtype=float)
    if freqs_ghz is None or freqs_ghz.size == 0:
        raise ValueError("raw must contain 'freqs_ghz'.")

    # pull counts
    if "sig_counts" in raw and "ref_counts" in raw:
        sig = np.asarray(raw["sig_counts"], dtype=float)  # (runs, steps)
        ref = np.asarray(raw["ref_counts"], dtype=float)
    else:
        counts = np.asarray(raw["counts"], dtype=float)
        if counts.shape[0] != 2:
            raise ValueError(f"Expected raw['counts'] with first dim=2 (sig/ref). Got {counts.shape}.")
        sig = counts[0]
        ref = counts[1]

    if sig.shape != ref.shape:
        raise ValueError(f"sig/ref shapes differ: {sig.shape} vs {ref.shape}")

    # per-run normalization (avoid div0)
    eps = 1e-12
    norm = sig / np.maximum(ref, eps)              # (runs, steps)
    norm_mean = np.nanmean(norm, axis=0)
    norm_sem = _sem(norm, axis=0)

    # contrast: resonance dip => positive peak
    contrast = 1.0 - norm
    contrast_mean = np.nanmean(contrast, axis=0)
    contrast_sem = _sem(contrast, axis=0)

    # baseline and peak
    baseline_norm = _edges_baseline(norm_mean, frac=baseline_edge_frac)
    contrast_peak = float(np.nanmax(_moving_average(contrast_mean, smooth_w)))
    f0_guess_ghz, idx_peak = _initial_resonance_from_contrast(freqs_ghz, contrast_mean, smooth_w=smooth_w)
    fwhm_guess_mhz = _rough_fwhm(freqs_ghz, contrast_mean, idx_peak, frac=0.5)

    # detuning axis (MHz) relative to guess or center
    center_ghz = float(raw.get("center_freq_ghz", np.nan))
    if np.isfinite(center_ghz):
        detuning_mhz = (freqs_ghz - center_ghz) * 1e3
    else:
        detuning_mhz = (freqs_ghz - f0_guess_ghz) * 1e3

    # --- Fit Lorentzian dip (optional SciPy) ---
    fit = {
        "success": False,
        "params": None,
        "cov": None,
        "f0_ghz": f0_guess_ghz,
        "fwhm_mhz": fwhm_guess_mhz,
        "contrast_peak": contrast_peak,
    }

    # Model on norm_mean: y = y0 - A / (1 + ((f-f0)/gamma)^2)
    def lorentz_dip(f, y0, A, f0, gamma):
        return y0 - A / (1.0 + ((f - f0) / gamma) ** 2)

    try:
        from scipy.optimize import curve_fit  # type: ignore

        y = norm_mean
        f = freqs_ghz

        y0_0 = float(_edges_baseline(y, frac=baseline_edge_frac))
        # A approx: dip depth
        dip_depth0 = float(np.nanmax([0.0, y0_0 - np.nanmin(y)]))
        A0 = dip_depth0

        f0_0 = f0_guess_ghz
        # gamma from rough fwhm: FWHM = 2*gamma
        gamma0 = (fwhm_guess_mhz * 1e-3) / 2.0 if np.isfinite(fwhm_guess_mhz) and fwhm_guess_mhz > 0 else (np.ptp(f) / 20.0)
        gamma0 = float(max(gamma0, 1e-6))

        p0 = [y0_0, A0, f0_0, gamma0]
        bounds = (
            [0.0, 0.0, np.min(f), 1e-7],
            [np.inf, np.inf, np.max(f), np.inf],
        )

        popt, pcov = curve_fit(lorentz_dip, f, y, p0=p0, bounds=bounds, maxfev=20000)

        y0_fit, A_fit, f0_fit, gamma_fit = [float(x) for x in popt]
        fwhm_fit_mhz = float(2.0 * gamma_fit * 1e3)  # GHz -> MHz
        contrast_fit = float(A_fit / max(y0_fit, 1e-12))  # fractional (approx)

        fit.update(
            {
                "success": True,
                "params": {"y0": y0_fit, "A": A_fit, "f0_ghz": f0_fit, "gamma_ghz": gamma_fit},
                "cov": pcov,
                "f0_ghz": f0_fit,
                "fwhm_mhz": fwhm_fit_mhz,
                "contrast_peak": contrast_fit,
            }
        )

    except Exception:
        # no SciPy or fit failed; we keep guesses
        pass

    # --- Plot ---
    fig = None
    if do_plot:
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = plt.gca()

            ax.errorbar(detuning_mhz, norm_mean, yerr=norm_sem, fmt="o", markersize=3, capsize=2)
            ax.set_xlabel("Detuning (MHz)")
            ax.set_ylabel("Normalized signal (sig/ref)")

            if title is None:
                nv_name = raw.get("nv_sig", None)
                title = "Confocal ESR"
            ax.set_title(title)

            # overlay fit if available
            if fit["success"] and fit["params"] is not None:
                f_dense = np.linspace(freqs_ghz.min(), freqs_ghz.max(), 800)
                if np.isfinite(center_ghz):
                    det_dense = (f_dense - center_ghz) * 1e3
                else:
                    det_dense = (f_dense - fit["f0_ghz"]) * 1e3

                p = fit["params"]
                y_dense = lorentz_dip(f_dense, p["y0"], p["A"], p["f0_ghz"], p["gamma_ghz"])
                ax.plot(det_dense, y_dense, "-", linewidth=2)

            ax.grid(True, alpha=0.2)

            # annotate
            f0_ghz = float(fit["f0_ghz"])
            fwhm_mhz = float(fit["fwhm_mhz"])
            ax.text(
                0.02,
                0.98,
                f"f0 = {f0_ghz:.6f} GHz\nFWHM ≈ {fwhm_mhz:.2f} MHz",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
            )

            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
        except Exception:
            fig = None

    results = {
        "freqs_ghz": freqs_ghz,
        "detuning_mhz": detuning_mhz,
        "norm_per_run": norm,
        "norm_mean": norm_mean,
        "norm_sem": norm_sem,
        "contrast_mean": contrast_mean,
        "contrast_sem": contrast_sem,
        "baseline_norm": baseline_norm,
        "f0_guess_ghz": f0_guess_ghz,
        "fwhm_guess_mhz": fwhm_guess_mhz,
        "fit": fit,
        "fig": fig,
    }
    return results

# -----------------------------
# small helpers (same style as your rabi file)
# -----------------------------
def _nv_get(nv_sig, key, default=None):
    if hasattr(nv_sig, key):
        return getattr(nv_sig, key)
    try:
        return nv_sig.get(key, default)
    except Exception:
        return default


def _get_nv_name(nv_sig) -> str:
    name = _nv_get(nv_sig, "name", None)
    return str(name) if name is not None else "confocal_nv"


def _as_single_int_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != 1:
            raise ValueError(f"Confocal ESR expects exactly one uwave channel; got {x!r}")
        return [int(x[0])]
    return [int(x)]


def _vkey_to_arg(vkey) -> str:
    if isinstance(vkey, VirtualLaserKey):
        return vkey.name
    if isinstance(vkey, str):
        return vkey.split(".")[-1]
    raise TypeError(f"Bad virtual laser key: {vkey!r}")


def _pd_lookup_ns(pulse_durations: dict, vkey: VirtualLaserKey):
    if not pulse_durations:
        return None
    candidates = [vkey, vkey.name, vkey.name.lower()]
    if vkey is VirtualLaserKey.CHARGE_POL:
        candidates += ["charge_pol"]
    if vkey is VirtualLaserKey.SPIN_READOUT:
        candidates += ["spin_readout", "spin_readout_dur"]
    for k in candidates:
        if k in pulse_durations and pulse_durations[k] is not None:
            return int(pulse_durations[k])
    return None


def _get_duration_ns(nv_sig, vkey: VirtualLaserKey, cfg, cfg_fallback_key_vkey: VirtualLaserKey) -> int:
    pd = _nv_get(nv_sig, "pulse_durations", {}) or {}
    v = _pd_lookup_ns(pd, vkey)
    if v is not None:
        return int(v)
    return int(cfg["Optics"]["VirtualLasers"][cfg_fallback_key_vkey]["duration"])


def build_base_args(
    nv_sig,
    *,
    pol_ns=None,
    readout_ns=None,
    uwave_ind=0,
    readout_vkey=VirtualLaserKey.SPIN_READOUT,
    readout_power=None,
    max_freq_ghz=3.2,  # optional metadata for your seq, if you want it
):
    """
    Keep the same general pattern as Rabi:
      base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_freq_ghz]
    """
    cfg = common.get_config_dict()

    if pol_ns is None:
        pol_ns = _get_duration_ns(nv_sig, VirtualLaserKey.CHARGE_POL, cfg, VirtualLaserKey.CHARGE_POL)
    if readout_ns is None:
        readout_ns = _get_duration_ns(nv_sig, VirtualLaserKey.SPIN_READOUT, cfg, VirtualLaserKey.SPIN_READOUT)

    uwave_ind_list = _as_single_int_list(uwave_ind)
    readout_vkey_arg = _vkey_to_arg(readout_vkey)

    return [
        int(pol_ns),
        int(readout_ns),
        uwave_ind_list,
        readout_vkey_arg,
        readout_power,
        float(max_freq_ghz),
    ]


def main(
    nv_sig,
    *,
    center_freq_ghz=2.8786,
    span_mhz=50.0,          # total span (e.g. 50 MHz sweep => ±25 MHz)
    num_steps=101,
    num_reps=20000,
    num_runs=6,
    uwave_ind=0,
    pol_ns=None,
    readout_ns=None,
    readout_vkey=VirtualLaserKey.SPIN_READOUT,
    readout_power=None,
    apd_indices=(0,),
    norm_mode=NormMode.SINGLE_VALUED,
    do_save=True,
):
    kpl.init_kplotlib()

    # build frequency axis in GHz
    span_ghz = float(span_mhz) * 1e-3  # MHz -> GHz
    fmin = float(center_freq_ghz) - 0.5 * span_ghz
    fmax = float(center_freq_ghz) + 0.5 * span_ghz
    freqs_ghz = np.linspace(fmin, fmax, int(num_steps)).astype(float)

    base_args = build_base_args(
        nv_sig,
        pol_ns=pol_ns,
        readout_ns=readout_ns,
        uwave_ind=uwave_ind,
        readout_vkey=readout_vkey,
        readout_power=readout_power,
        max_freq_ghz=float(np.max(freqs_ghz)),
    )

    print(f"[ESR] center={center_freq_ghz:.6f} GHz, span={span_mhz:.3f} MHz, steps={num_steps}")
    print(f"[ESR base_args] pol_ns={base_args[0]}, readout_ns={base_args[1]}")

    seq_file = "esr_seq.py"

    def step_fn(step_ind: int):
        f_ghz = float(freqs_ghz[int(step_ind)])

        # If your esr_seq.py only needs base_args:
        seq_args = [base_args]
        # If it also wants current freq, you can pass it too:
        # seq_args = [base_args, f_ghz]

        seq_args_string = tb.encode_seq_args(seq_args)

        # IMPORTANT: return THIRD value = per-step freq (GHz)
        return seq_file, seq_args_string, f_ghz

    raw = base.main(
        nv_sig=nv_sig,
        num_steps=int(num_steps),
        num_reps=int(num_reps),
        num_runs=int(num_runs),
        run_fn=None,
        step_fn=step_fn,
        uwave_ind_list=[int(uwave_ind)],
        num_exps=2,                      # sig + ref
        apd_indices=list(apd_indices),
        load_iq=False,
        stream_load_in_run_fn=False,
        charge_prep_fn=None,
    )

    # counts: (num_exps, num_runs, num_steps) aggregated
    counts = np.asarray(raw["counts"])
    if counts.shape[0] != 2:
        raise RuntimeError(f"Expected 2 experiments (sig/ref). Got counts.shape={counts.shape}")

    sig_counts = counts[0].astype(int)  # (runs, steps)
    ref_counts = counts[1].astype(int)

    # Optionally compute norm here (same as your rabi postprocess pattern)
    # readout_ns_eff = int(base_args[1])
    # sig_kcps, ref_kcps, norm, norm_ste = tb.process_counts(
    #     sig_counts, ref_counts, int(num_reps), readout_ns_eff, norm_mode=norm_mode
    # )

    raw.update(
        {
            "center_freq_ghz": float(center_freq_ghz),
            "span_mhz": float(span_mhz),
            "freqs_ghz": freqs_ghz.tolist(),
            "freq-units": "GHz",
            "base_args": base_args,
            "sig_counts": sig_counts.tolist(),
            "ref_counts": ref_counts.tolist(),
            # "norm": np.asarray(norm).tolist(),
            # "norm_ste": np.asarray(norm_ste).tolist(),
            # "norm_mode": str(norm_mode),
        }
    )

    if do_save:
        try:
            timestamp = dm.get_time_stamp()
            nv_name = _get_nv_name(nv_sig)
            file_path = dm.get_file_path(__file__, timestamp, nv_name)
            raw["timestamp"] = timestamp
            dm.save_raw_data(raw, file_path)
            print(f"[ESR] saved: {file_path}")
        except Exception:
            print("[SAVE ERROR]\n", traceback.format_exc())

    res = analyze_esr(
        raw,
        smooth_w=7,
        do_plot=True,
        save_path=None,  # or "C:/.../esr.png"
        title="ESR around 2.8786 GHz",
    )

    print("Fit success:", res["fit"]["success"])
    print("f0 (GHz):", res["fit"]["f0_ghz"])
    print("FWHM (MHz):", res["fit"]["fwhm_mhz"])
    print("Contrast:", res["fit"]["contrast_peak"])
    
    tb.reset_cfm()
    return raw


if __name__ == "__main__":

    file_stem = "2026_01_07-13_49_03-(Wu)"
    # raw, proc, fit, fig, res = load_and_analyze_esr(
    #     file_stem,
    #     smooth_w=7,
    #     do_plot=True,
    #     save_path=None,
    #     title="ESR around 2.8786 GHz",
    # )

    # print("Fit success:", res["fit"]["success"])
    # print("f0 (GHz):", res["fit"]["f0_ghz"])
    # print("FWHM (MHz):", res["fit"]["fwhm_mhz"])
    # print("Contrast:", res["fit"]["contrast_peak"])
