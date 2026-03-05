# -*- coding: utf-8 -*-
"""
majorroutines/confocal/confocal_rabi.py

Confocal Rabi using the unified confocal_base_routine + SWAB_82 pulse-gen + SWAB_20 tagger.

Compatible with your rabi_seq.py contract:

  seq_args = [base_args, tau]          (or [base_args, tau, num_reps_ignored])
  base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau_ns]

And rabi_seq produces EXACTLY 2 APD gates per repetition:
  gate0 = signal readout
  gate1 = reference readout

@author: Saroj Chand
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.confocal.confocal_base_routine as base
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import NormMode, VirtualLaserKey


def _as_array(x):
    # raw saved data may store lists; normalize to ndarray
    return np.asarray(x)


def _counts_to_runs_steps(counts):
    counts = _as_array(counts)

    if counts.ndim == 4:
        # (2, runs, steps, reps) -> (2, runs, steps)
        return counts.sum(axis=-1)
    if counts.ndim == 3:
        # (2, runs, steps)
        return counts
    raise RuntimeError(f"Unexpected counts shape: {counts.shape}")


def _get_readout_ns(raw):
    # Prefer base_args[1] if present
    if "base_args" in raw and raw["base_args"] is not None:
        try:
            return int(_as_array(raw["base_args"])[1])
        except Exception:
            pass
    # fallbacks
    for k in ("readout_ns", "readout_ns_eff"):
        if k in raw and raw[k] is not None:
            return int(raw[k])
    raise KeyError("Could not infer readout_ns from raw (need base_args[1] or readout_ns).")


def _get_taus_ns(raw):
    if "taus_ns" in raw and raw["taus_ns"] is not None:
        return _as_array(raw["taus_ns"]).astype(float)
    # fallback: reconstruct if min/max/num_steps were saved
    for keys in (("tau_min", "tau_max", "num_steps"), ("min_tau", "max_tau", "num_steps")):
        if all(k in raw for k in keys):
            tau_min, tau_max, num_steps = (raw[keys[0]], raw[keys[1]], raw[keys[2]])
            return np.linspace(int(tau_min), int(tau_max), int(num_steps)).astype(float)
    raise KeyError("Could not infer taus_ns from raw (save taus_ns, or tau_min/tau_max/num_steps).")


def postprocess_raw_rabi(
    raw: dict,
    *,
    norm_mode=NormMode.SINGLE_VALUED,
    force_simple_norm=False,
):
    """
    Returns a dict with:
      taus_ns
      sig_counts, ref_counts: (runs, steps) aggregated
      sig_kcps, ref_kcps: (steps,)
      norm, norm_ste: (steps,)
      norm_per_run: (runs, steps)
    """
    counts_rs = _counts_to_runs_steps(raw["counts"])  # (2, runs, steps, RE)

    if counts_rs.shape[0] != 2:
        raise RuntimeError(f"Expected 2 experiments (sig/ref). Got {counts_rs.shape}")

    sig_counts = counts_rs[0].astype(float)  # (runs, steps)
    ref_counts = counts_rs[1].astype(float)

    num_reps = int(raw["num_reps"])
    readout_ns = int(_get_readout_ns(raw))
    taus_ns = _get_taus_ns(raw)

    # Preferred: use your existing helper (handles kcps + error properly)
    if (not force_simple_norm) and hasattr(tb, "process_counts"):
        sig_kcps, ref_kcps, norm, norm_ste = tb.process_counts(
            sig_counts,
            ref_counts,
            int(num_reps),
            int(readout_ns),
            norm_mode=norm_mode,
        )
        norm_per_run = None  # tb.process_counts may not expose it
    else:
        # Simple fallback: norm = mean(sig/ref) across runs, with run-to-run STE
        eps = 1e-12
        norm_per_run = sig_counts / (ref_counts + eps)
        norm = np.nanmean(norm_per_run, axis=0)
        norm_ste = np.nanstd(norm_per_run, axis=0, ddof=1) / np.sqrt(norm_per_run.shape[0])

        # kcps (mean over runs)
        sig_mean = np.nanmean(sig_counts, axis=0)
        ref_mean = np.nanmean(ref_counts, axis=0)
        sig_kcps = (sig_mean / num_reps) / (readout_ns * 1e-9) / 1e3
        ref_kcps = (ref_mean / num_reps) / (readout_ns * 1e-9) / 1e3

    return dict(
        taus_ns=taus_ns,
        readout_ns=readout_ns,
        num_reps=num_reps,
        sig_counts=sig_counts,
        ref_counts=ref_counts,
        sig_kcps=_as_array(sig_kcps),
        ref_kcps=_as_array(ref_kcps),
        norm=_as_array(norm),
        norm_ste=_as_array(norm_ste),
        norm_mode=str(norm_mode),
        norm_per_run=norm_per_run,
    )


def plot_rabi(taus_ns, norm, norm_ste=None, title="Confocal Rabi (postprocess)"):
    fig, ax = plt.subplots()
    kpl.plot_points(ax, taus_ns, norm, norm_ste, label="data")
    ax.set_xlabel("MW pulse length τ (ns)")
    ax.set_ylabel("Normalized signal")
    ax.set_title(title)
    ax.legend()
    return fig


def fit_rabi_basic(taus_ns, norm, norm_ste=None):
    """
    Fit using tb.cosexp_1_at_0(t, offset, freq, decay) if available.
    Returns dict with popt, red_chi_sq, rabi_period_ns.
    """
    if not hasattr(tb, "curve_fit") or not hasattr(tb, "cosexp_1_at_0"):
        return {"popt": None, "pcov": None, "red_chi_sq": None, "rabi_period_ns": None}

    t = np.asarray(taus_ns, dtype=float)
    y = np.asarray(norm, dtype=float)
    if norm_ste is None:
        sigma = np.full_like(y, 1e-3)
    else:
        sigma = np.asarray(norm_ste, dtype=float)
        sigma = np.where((sigma <= 0) | (~np.isfinite(sigma)), np.nanmedian(sigma[sigma > 0]), sigma)

    # crude FFT guess for freq (cycles/ns)
    y0 = y - np.nanmean(y)
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1.0
    freqs = np.fft.rfftfreq(len(y0), d=dt)
    spec = np.abs(np.fft.rfft(np.nan_to_num(y0)))
    if len(spec) > 2:
        i = int(np.argmax(spec[1:]) + 1)
        freq_guess = float(freqs[i])
    else:
        freq_guess = 1.0 / max(1.0, (t[-1] - t[0]))

    offset_guess = float(np.nanmean(y[-max(3, len(y)//5):]))
    decay_guess = float(max(50.0, 0.5 * (t[-1] - t[0])))

    p0 = [offset_guess, max(freq_guess, 1e-6), decay_guess]
    popt, pcov, red_chi_sq = tb.curve_fit(tb.cosexp_1_at_0, t, y, p0=p0, sigma=sigma)

    rabi_period_ns = None
    if popt is not None and popt[1] not in (0, None):
        rabi_period_ns = float(1.0 / popt[1])

    return {"popt": popt, "pcov": pcov, "red_chi_sq": red_chi_sq, "rabi_period_ns": rabi_period_ns}


def load_and_analyze(
    *,
    file_stem=None,
    do_fit=True,
    do_plot=True,
    save_analysis=True,
):
    """
    Main entry point.

    Usage:
      load_and_analyze(file_id=1234567890123)
      load_and_analyze(file_path=".../confocal_rabi_....npz")
    """
    kpl.init_kplotlib()
    raw = dm.get_raw_data(file_stem=file_stem, load_npz=True, use_cache=True)
    # counts = raw["counts"]
    # print(counts.shape)
    # print(f"[postprocess] loaded raw data with keys: {list(raw.keys())}")
    proc = postprocess_raw_rabi(
        raw,
        norm_mode=NormMode.SINGLE_VALUED,
        force_simple_norm=False,
    )
    fit = fit_rabi_basic(proc["taus_ns"], proc["norm"], proc["norm_ste"])

    fig = None
    if do_plot:
        title = "Confocal Rabi (postprocess)"
        fig = plot_rabi(proc["taus_ns"], proc["norm"], proc["norm_ste"], title=title)
        if fit and fit.get("popt", None) is not None:
            tfit = np.linspace(np.min(proc["taus_ns"]), np.max(proc["taus_ns"]), 800)
            yfit = tb.cosexp_1_at_0(tfit, *fit["popt"])
            kpl.plot_line(fig.axes[0], tfit, yfit, label="fit")
            fig.axes[0].legend()
        kpl.show()

    # Optionally save analysis next to raw
    if save_analysis:
        try:
            out = dict(raw)
            out.update(
                {
                    "analysis_norm": proc["norm"],
                    "analysis_norm_ste": proc["norm_ste"],
                    "analysis_sig_kcps": proc["sig_kcps"],
                    "analysis_ref_kcps": proc["ref_kcps"],
                }
            )
            if fit is not None:
                out.update(
                    {
                        "analysis_fit_popt": None if fit["popt"] is None else np.asarray(fit["popt"]),
                        "analysis_fit_red_chi_sq": fit["red_chi_sq"],
                        "analysis_fit_rabi_period_ns": fit["rabi_period_ns"],
                    }
                )

            # save as a sibling “-analysis” file
            ts = raw.get("timestamp", dm.get_time_stamp())
            nv_name = raw.get("nv_name", raw.get("nv_sig", "confocal_nv"))
            nv_name = str(nv_name)

            analysis_path = dm.get_file_path(__file__, ts, nv_name + "-analysis")
            dm.save_raw_data(out, analysis_path)

            if fig is not None:
                dm.save_figure(fig, analysis_path)

            print(f"[postprocess] saved analysis to: {analysis_path}")
        except Exception as e:
            print("[postprocess] save_analysis failed:", repr(e))

    return raw, proc, fit, fig


# -----------------------------
# small helpers (robust nv_sig)
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
            raise ValueError(f"Confocal Rabi expects exactly one uwave channel; got {x!r}")
        return [int(x[0])]
    return [int(x)]


def _vkey_to_arg(vkey) -> str:
    """
    rabi_seq.py accepts:
      - "SPIN_READOUT"
      - "VirtualLaserKey.SPIN_READOUT"
    We'll send "SPIN_READOUT" (json-friendly).
    """
    if isinstance(vkey, VirtualLaserKey):
        return vkey.name
    if isinstance(vkey, str):
        return vkey.split(".")[-1]
    raise TypeError(f"Bad virtual laser key: {vkey!r}")


def _pd_lookup_ns(pulse_durations: dict, vkey: VirtualLaserKey):
    """
    Try common key variants in nv_sig.pulse_durations:
      - Enum key
      - name string
      - lower-case name
      - a couple legacy strings
    """
    if not pulse_durations:
        return None

    candidates = [vkey, vkey.name, vkey.name.lower()]

    # legacy-ish fallbacks
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

    # fallback to config durations
    return int(cfg["Optics"]["VirtualLasers"][cfg_fallback_key_vkey]["duration"])

# -----------------------------
# base_args builder (IMPORTANT)
# -----------------------------
def build_base_args(
    nv_sig,
    *,
    pol_ns=None,
    readout_ns=None,
    uwave_ind=0,
    readout_vkey=VirtualLaserKey.SPIN_READOUT,
    readout_power=None,
    max_tau_ns=400,
):
    """
    MUST match rabi_seq.py:
      base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau_ns]
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
        uwave_ind_list,          # list[int]
        readout_vkey_arg,        # "SPIN_READOUT"
        readout_power,           # None OK for DIGITAL lasers
        int(max_tau_ns),
    ]
# -----------------------------
# main
# -----------------------------
def main(
    nv_sig,
    *,
    num_steps=51,
    num_reps=20000,
    num_runs=6,
    tau_min=0,
    tau_max=400,
    uwave_ind=0,
    pol_ns=None,
    readout_ns=None,
    readout_vkey=VirtualLaserKey.SPIN_READOUT,
    readout_power=None,
    apd_indices=(0,),
    norm_mode=NormMode.SINGLE_VALUED,
    do_plot=True,
    do_save=True,
):
    kpl.init_kplotlib()

    taus = np.linspace(int(tau_min), int(tau_max), int(num_steps)).astype(np.int32)
    max_tau_ns = int(np.max(taus))

    base_args = build_base_args(
        nv_sig,
        pol_ns=pol_ns,
        readout_ns=readout_ns,
        uwave_ind=uwave_ind,
        readout_vkey=readout_vkey,
        readout_power=readout_power,
        max_tau_ns=max_tau_ns,
    )

    # nice one-time print
    print(f"[RABI base_args] pol_ns={base_args[0]}, readout_ns={base_args[1]}, max_tau={base_args[5]}")

    seq_file = "rabi_seq.py"

    def step_fn(step_ind: int):
        tau = int(taus[step_ind])

        # IMPORTANT: ONLY return args; base routine does stream_load + run.
        seq_args = [base_args, tau]  # (or [base_args, tau, num_reps_ignored])
        seq_args_string = tb.encode_seq_args(seq_args)

        # Debug should look like:
        # [[10000, 300, [0], "SPIN_READOUT", None, 400], 376]
        # print("SEQ ARGS STRING:", seq_args_string)

        return seq_file, seq_args_string

    raw = base.main(
        nv_sig=nv_sig,
        num_steps=int(num_steps),
        num_reps=int(num_reps),
        num_runs=int(num_runs),
        run_fn=None,
        step_fn=step_fn,
        uwave_ind_list=[int(uwave_ind)],
        num_exps=2,                 # sig + ref
        apd_indices=list(apd_indices),
        load_iq=False,
        stream_load_in_run_fn=False,
        charge_prep_fn=None,
    )

    # -----------------------------
    # process counts (expected shape)
    # counts: (num_exps, num_runs, num_steps, num_reps)
    # -----------------------------
    counts = np.asarray(raw["counts"])
    if counts.shape[0] != 2:
        raise RuntimeError(f"Expected 2 experiments (sig/ref). Got counts.shape={counts.shape}")

    sig_counts = counts[0].sum(axis=-1)  # (runs, steps)
    ref_counts = counts[1].sum(axis=-1)  # (runs, steps)

    readout_ns_eff = int(base_args[1])
    # sig_kcps, ref_kcps, norm, norm_ste = tb.process_counts(
    #     sig_counts,
    #     ref_counts,
    #     int(num_reps),
    #     readout_ns_eff,
    #     norm_mode=norm_mode,
    # )

    raw.update(
        {
            "taus_ns": taus.astype(int).tolist(),
            "tau-units": "ns",
            "base_args": base_args,
            "sig_counts_sum": sig_counts.astype(int).tolist(),
            "ref_counts_sum": ref_counts.astype(int).tolist(),
            # "sig_kcps": np.asarray(sig_kcps).tolist(),
            # "ref_kcps": np.asarray(ref_kcps).tolist(),
            # "norm": np.asarray(norm).tolist(),
            # "norm_ste": np.asarray(norm_ste).tolist(),
            # "norm_mode": str(norm_mode),
        }
    )

    # -----------------------------
    # save + plot
    # -----------------------------
    fig = None
    file_path = None
    timestamp = None

    if do_save:
        try:
            timestamp = dm.get_time_stamp()
            nv_name = _get_nv_name(nv_sig)
            file_path = dm.get_file_path(__file__, timestamp, nv_name)
            raw["timestamp"] = timestamp
            dm.save_raw_data(raw, file_path)
        except Exception:
            print("[SAVE ERROR]\n", traceback.format_exc())

    # if do_plot:
    #     try:
    #         title = f"Confocal Rabi: {_get_nv_name(nv_sig)}"
    #         fig = _plot_rabi(taus, norm, norm_ste, title=title)
    #         kpl.show()
    #     except Exception:
    #         print("[PLOT ERROR]\n", traceback.format_exc())

    if do_save and (fig is not None) and (file_path is not None):
        try:
            dm.save_figure(fig, file_path)
        except Exception:
            print("[FIG SAVE ERROR]\n", traceback.format_exc())

    tb.reset_cfm()
    return raw


if __name__ == "__main__":

    file_stem = "2026_01_07-13_49_03-(Wu)"
    raw, proc, fit, fig = load_and_analyze(file_stem=file_stem)
    # or:
    # raw, proc, fit, fig = load_and_analyze(file_path=r"C:\...\your_saved_file.npz")
    kpl.show(block=True)
