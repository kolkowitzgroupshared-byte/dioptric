# -*- coding: utf-8 -*-
"""
Single resonance measurement without base routine.

Very close to old working ESR style:
- stream_load once
- set microwave frequency
- stream_start
- read_counter_modulo_gates(2, 1)

Returns:
    raw_data, proc_data
"""

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode


def main(
    nv_sig,
    freq_center_ghz=2.8786,
    freq_span_mhz=200.0,
    num_steps=51,
    num_reps=1,
    num_runs=20,
    uwave_ind=0,
    readout_vkey=VirtualLaserKey.IMAGING,
    readout_ns=None,
    uwave_power_dbm=None,
    laser_power=None,
    do_targeting=True,
    do_plot=True,
    do_save=True,
    shuffle=False,
    norm_mode=NormMode.SINGLE_VALUED,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()
        
    # readout_vkey=VirtualLaserKey.SPIN_READOUT
    readout_vkey=VirtualLaserKey.if hasattr(nv_sig, "readout_vkey"):

    vld = tb.get_virtual_laser_dict(readout_vkey)
    if readout_ns is None:
        readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld["duration"])))
    readout_ns = int(readout_ns)

    # MW setup
    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)
    if uwave_power_dbm is not None:
        sig_gen.set_amp(float(uwave_power_dbm))
    sig_gen.set_freq(float(freq_ghz))
    sig_gen.uwave_on()

    # Sequence setup
    seq_file = "resonance.py"
    seq_args = [
        # pol_ns,
        readout_ns,
        int(uwave_ind),
        readout_vkey.name if hasattr(readout_vkey, "name") else str(readout_vkey),
        laser_power,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)

    pulsegen_server.stream_load(seq_file, seq_args_string)

    # frequency axis
    span_ghz = freq_span_mhz * 1e-3
    freqs_ghz = np.linspace(
        freq_center_ghz - span_ghz / 2,
        freq_center_ghz + span_ghz / 2,
        num_steps,
    )

    sweep_order = np.arange(num_steps)
    if shuffle:
        np.random.shuffle(sweep_order)

    sig_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ref_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()

    # plotting
    if do_plot:
        fig, ax = plt.subplots()
        ax.set_xlabel("Run")
        ax.set_ylabel("Counts")
        (line_ref,) = ax.plot([], [], label="Ref")
        (line_sig,) = ax.plot([], [], label="Sig")
        ax.legend()
    else:
        fig = None

    tb.init_safe_stop()

    for run_ind in range(num_runs):
        counter_server.start_tag_stream()

        print(f"Run {run_ind + 1}/{num_runs}")
        # print(f"uwave_power: {uwave_power_dbm}")
        # print(f" VirtualLaserKey.SPIN_READOUT: {readout_ns} ns")


        if tb.safe_stop():
            break

        if do_targeting:
            try:
                opti_coords = targeting.main_with_cxn(nv_sig)
                opti_coords_list.append(opti_coords)
            except Exception as e:
                print(f"Targeting failed on run {run_ind}: {e}")
                opti_coords_list.append(None)

        
        try:
            for step_ind in sweep_order:
                if tb.safe_stop():
                    break

                f = float(freqs_ghz[step_ind])
                sig_gen.set_freq(f)

                counter_server.clear_buffer()
                pulsegen_server.stream_start(int(num_reps))

                new_counts = counter_server.read_counter_modulo_gates(2, 1)
                sample_counts = new_counts[0]
                # print("len(new_counts) =", len(new_counts))
                # print("first few =", new_counts[:5])

                # gate0 = ref, gate1 = sig
                ref_counts[run_ind, step_ind] = sample_counts[0]
                sig_counts[run_ind, step_ind] = sample_counts[1]

        finally:
            try:
                counter_server.stop_tag_stream()
            except Exception:
                pass

        if do_plot:
            valid_runs = np.isfinite(sig_counts[: run_ind + 1]) & np.isfinite(ref_counts[: run_ind + 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_runs = sig_counts[: run_ind + 1] / np.maximum(ref_counts[: run_ind + 1], 1)
            norm_mean = np.nanmean(norm_runs, axis=0)

            line.set_data(freqs_ghz, norm_mean)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        if do_save:
            raw_incremental = {
                "timestamp": timestamp,
                "nv_sig": nv_sig,
                "freq_ghz": float(freq_ghz),
                "num_reps": int(num_reps),
                "num_runs": int(num_runs),
                "uwave_ind": int(uwave_ind),
                "uwave_power_dbm": uwave_power_dbm,
                "readout_ns": int(readout_ns),
                "opti_coords_list": opti_coords_list,
                "sig_counts": sig_counts.tolist(),
                "ref_counts": ref_counts.tolist(),
            }
            file_path = dm.get_file_path(
                __file__, timestamp, nv_sig["name"], "incremental"
            )
            dm.save_raw_data(raw_incremental, file_path)

    # process
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_runs = sig_counts / np.maximum(ref_counts, 1)

    norm_mean = np.nanmean(norm_runs, axis=0)
    norm_ste = np.nanstd(norm_runs, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(norm_runs), axis=0))

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "freq_ghz": float(freq_ghz),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_power_dbm": uwave_power_dbm,
        "readout_ns": int(readout_ns),
        "opti_coords_list": opti_coords_list,
        "sig_counts": sig_counts.tolist(),
        "ref_counts": ref_counts.tolist(),
        "sig_kcps": sig_kcps,
        "ref_kcps": ref_kcps,
        "norm": norm,
        "norm_ste": norm_ste,
    }

    print("\nFinal results")
    print(f"freq       = {freq_ghz:.6f} GHz")
    print(f"sig kcps   = {sig_kcps:.3f}")
    print(f"ref kcps   = {ref_kcps:.3f}")
    print(f"norm       = {norm:.6f} ± {norm_ste:.6f}")
    print(f"contrast   = {proc_data['contrast']:.6f}")

    if do_save:
        file_path = dm.get_file_path(__file__, timestamp, nv_sig["name"])
        if fig is not None:
            dm.save_figure(fig, file_path)
        dm.save_raw_data(raw_data, file_path)

    tb.reset_cfm()
    return raw_data, proc_data


if __name__ == "__main__":
    # example:
    # raw, proc = main(
    #     nv_sig=nv_sig,
    #     freq_ghz=2.8786,
    #     num_reps=10000,
    #     num_runs=10,
    #     uwave_ind=0,
    # )
    pass

# # -*- coding: utf-8 -*-
# """
# Confocal ESR acquisition + analysis.

# - Sweeps sig-gen frequency each step (Python)
# - Reference is MW OFF inside sequence (2 APD gates per rep)
# - Uses base.main(... num_exps=2 ...)
# """

# import os
# import traceback
# import numpy as np

# import majorroutines.confocal.confocal_base_routine as base
# from utils import common
# from utils import data_manager as dm
# from utils import kplotlib as kpl
# from utils import tool_belt as tb
# from utils.constants import VirtualLaserKey


# # --------------------------
# # Analysis helpers
# # --------------------------
# def _sem(x, axis=0):
#     x = np.asarray(x, dtype=float)
#     n = np.sum(np.isfinite(x), axis=axis)
#     return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


# def _moving_average(y, w=5):
#     y = np.asarray(y, dtype=float)
#     if w is None or w <= 1:
#         return y
#     w = int(w)
#     if w >= len(y):
#         return np.full_like(y, np.nanmean(y))
#     kernel = np.ones(w) / w
#     return np.convolve(y, kernel, mode="same")


# def _edges_baseline(y, frac=0.15):
#     y = np.asarray(y, dtype=float)
#     n = len(y)
#     k = max(1, int(round(frac * n)))
#     edges = np.concatenate([y[:k], y[-k:]])
#     return float(np.nanmedian(edges))


# def analyze_esr_2exp(
#     raw: dict,
#     *,
#     smooth_w: int = 7,
#     baseline_edge_frac: float = 0.15,
#     do_plot: bool = True,
#     save_path: str | None = None,
#     title: str | None = None,
# ):
#     """
#     Expects raw['counts'] shape (2, runs, steps) with:
#       counts[0] = MW ON (signal)
#       counts[1] = MW OFF (reference)
#     and raw['freqs_ghz'].
#     """
#     freqs_ghz = np.asarray(raw["freqs_ghz"], dtype=float)
#     counts = np.asarray(raw["counts"], dtype=float)

#     if counts.ndim != 3 or counts.shape[0] != 2:
#         raise ValueError(f"Expected counts shape (2, runs, steps). Got {counts.shape}")

#     sig = counts[0]  # (runs, steps)
#     ref = counts[1]

#     eps = 1e-12
#     norm = sig / np.maximum(ref, eps)
#     norm_mean = np.nanmean(norm, axis=0)
#     norm_sem = _sem(norm, axis=0)

#     contrast_mean = 1.0 - norm_mean  # ODMR dip -> positive peak

#     # peak estimate
#     idx = int(np.nanargmax(_moving_average(contrast_mean, smooth_w)))
#     f0_guess = float(freqs_ghz[idx])

#     # detuning axis
#     center_ghz = float(raw.get("center_freq_ghz", np.nan))
#     if np.isfinite(center_ghz):
#         detuning_mhz = (freqs_ghz - center_ghz) * 1e3
#     else:
#         detuning_mhz = (freqs_ghz - f0_guess) * 1e3

#     fit = {"success": False, "f0_ghz": f0_guess, "fwhm_mhz": np.nan, "contrast": float(np.nanmax(contrast_mean))}

#     # optional Lorentz dip fit on norm_mean
#     def lorentz_dip(f, y0, A, f0, gamma):
#         return y0 - A / (1.0 + ((f - f0) / gamma) ** 2)

#     try:
#         from scipy.optimize import curve_fit  # type: ignore

#         y = norm_mean
#         f = freqs_ghz

#         y0_0 = _edges_baseline(y, frac=baseline_edge_frac)
#         A0 = max(0.0, y0_0 - float(np.nanmin(y)))
#         gamma0 = max(1e-6, (np.ptp(f) / 30.0))
#         p0 = [y0_0, A0, f0_guess, gamma0]
#         bounds = ([0.0, 0.0, f.min(), 1e-7], [np.inf, np.inf, f.max(), np.inf])

#         popt, _pcov = curve_fit(lorentz_dip, f, y, p0=p0, bounds=bounds, maxfev=20000)
#         y0_fit, A_fit, f0_fit, gamma_fit = [float(v) for v in popt]
#         fit.update(
#             {
#                 "success": True,
#                 "f0_ghz": f0_fit,
#                 "fwhm_mhz": float(2.0 * gamma_fit * 1e3),
#                 "contrast": float(A_fit / max(y0_fit, 1e-12)),
#                 "params": {"y0": y0_fit, "A": A_fit, "f0_ghz": f0_fit, "gamma_ghz": gamma_fit},
#             }
#         )
#     except Exception:
#         pass

#     fig = None
#     if do_plot:
#         try:
#             import matplotlib.pyplot as plt

#             fig = plt.figure()
#             ax = plt.gca()
#             ax.errorbar(detuning_mhz, norm_mean, yerr=norm_sem, fmt="o", markersize=3, capsize=2)
#             ax.set_xlabel("Detuning (MHz)")
#             ax.set_ylabel("Normalized signal (MW on / MW off)")
#             ax.grid(True, alpha=0.2)

#             if title is None:
#                 title = "Confocal ESR (MW-on / MW-off ref)"
#             ax.set_title(title)

#             # overlay fit
#             if fit.get("success") and "params" in fit:
#                 f_dense = np.linspace(freqs_ghz.min(), freqs_ghz.max(), 800)
#                 if np.isfinite(center_ghz):
#                     det_dense = (f_dense - center_ghz) * 1e3
#                 else:
#                     det_dense = (f_dense - fit["f0_ghz"]) * 1e3
#                 p = fit["params"]
#                 ax.plot(det_dense, lorentz_dip(f_dense, p["y0"], p["A"], p["f0_ghz"], p["gamma_ghz"]), "-", linewidth=2)

#             ax.text(
#                 0.02,
#                 0.98,
#                 f"f0 = {fit['f0_ghz']:.6f} GHz\nFWHM ≈ {fit['fwhm_mhz']:.2f} MHz\ncontrast ≈ {fit['contrast']:.4f}",
#                 transform=ax.transAxes,
#                 va="top",
#                 ha="left",
#                 fontsize=10,
#             )

#             if save_path is not None:
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
#                 fig.savefig(save_path, dpi=300, bbox_inches="tight")
#         except Exception:
#             fig = None

#     return {
#         "freqs_ghz": freqs_ghz,
#         "detuning_mhz": detuning_mhz,
#         "norm_mean": norm_mean,
#         "norm_sem": norm_sem,
#         "fit": fit,
#         "fig": fig,
#     }


# # --------------------------
# # Small nv helpers
# # --------------------------
# def _nv_get(nv_sig, key, default=None):
#     if hasattr(nv_sig, key):
#         return getattr(nv_sig, key)
#     try:
#         return nv_sig.get(key, default)
#     except Exception:
#         return default


# def _get_nv_name(nv_sig) -> str:
#     name = _nv_get(nv_sig, "name", None)
#     return str(name) if name is not None else "confocal_nv"


# def _as_single_int_list(x):
#     if isinstance(x, (list, tuple, np.ndarray)):
#         if len(x) != 1:
#             raise ValueError(f"Confocal ESR expects exactly one uwave channel; got {x!r}")
#         return [int(x[0])]
#     return [int(x)]


# def _vkey_to_arg(vkey) -> str:
#     if isinstance(vkey, VirtualLaserKey):
#         return vkey.name
#     if isinstance(vkey, str):
#         return vkey.split(".")[-1]
#     raise TypeError(f"Bad virtual laser key: {vkey!r}")


# def build_base_args(
#     nv_sig,
#     *,
#     pol_ns: int,
#     readout_ns: int,
#     uwave_ind: int = 0,
#     readout_vkey=VirtualLaserKey.SPIN_READOUT,
#     readout_power=None,
#     max_mw_dur_ns: int = 2000,
# ):
#     uwave_ind_list = _as_single_int_list(uwave_ind)
#     return [
#         int(pol_ns),
#         int(readout_ns),
#         uwave_ind_list,
#         _vkey_to_arg(readout_vkey),
#         readout_power,
#         int(max_mw_dur_ns),
#     ]


# def main(
#     nv_sig,
#     *,
#     center_freq_ghz: float = 2.8786,
#     span_mhz: float = 40.0,
#     num_steps: int = 101,
#     num_reps: int = 20000,
#     num_runs: int = 6,
#     uwave_ind: int = 0,
#     uwave_power_dbm=None,     # if None: use config's stored uwave_power
#     mw_dur_ns: int = 2000,    # fixed MW ON duration for ESR
#     pol_ns: int = 10_000,
#     readout_ns: int = 300,
#     readout_vkey=VirtualLaserKey.SPIN_READOUT,
#     readout_power=None,
#     apd_indices=(0,),
#     shuffle_freqs: bool = True,
#     shuffle_seed: int = 0,
#     do_save: bool = True,
#     do_plot: bool = True,
# ):
#     kpl.init_kplotlib()

#     span_ghz = float(span_mhz) * 1e-3
#     fmin = float(center_freq_ghz) - 0.5 * span_ghz
#     fmax = float(center_freq_ghz) + 0.5 * span_ghz
#     freqs_ghz = np.linspace(fmin, fmax, int(num_steps)).astype(float)

#     order = np.arange(len(freqs_ghz))
#     if shuffle_freqs:
#         rng = np.random.default_rng(int(shuffle_seed))
#         rng.shuffle(order)
#     freqs_sweep = freqs_ghz[order]

#     base_args = build_base_args(
#         nv_sig,
#         pol_ns=pol_ns,
#         readout_ns=readout_ns,
#         uwave_ind=uwave_ind,
#         readout_vkey=readout_vkey,
#         readout_power=readout_power,
#         max_mw_dur_ns=int(mw_dur_ns),   # keep constant period
#     )

#     seq_file = "esr_seq.py"

#     # Set MW power once (optional)
#     sg = tb.get_server_sig_gen(int(uwave_ind))
#     sg_dict = tb.get_virtual_sig_gen_dict(int(uwave_ind))
#     if uwave_power_dbm is None:
#         uwave_power_dbm = sg_dict.get("uwave_power", None)
#     if uwave_power_dbm is not None:
#         sg.set_amp(float(uwave_power_dbm))

#     def step_fn(step_ind: int):
#         f_ghz = float(freqs_sweep[int(step_ind)])
#         sg.set_freq(f_ghz)
#         sg.uwave_on()

#     def run_fn(_step_ind: int):
#         seq_args = [base_args, int(mw_dur_ns)]
#         seq_args_string = tb.encode_seq_args(seq_args)
#         return seq_file, seq_args_string

#     raw = base.main(
#         nv_sig=nv_sig,
#         num_steps=int(len(freqs_sweep)),
#         num_reps=int(num_reps),
#         num_runs=int(num_runs),
#         run_fn=run_fn,
#         step_fn=step_fn,
#         uwave_ind_list=[int(uwave_ind)],
#         num_exps=2,
#         apd_indices=list(apd_indices),
#         load_iq=False,
#         stream_load_in_run_fn=False,
#         charge_prep_fn=None,
#     )

#     # Unshuffle back to ascending frequency for saving/plotting
#     counts = np.asarray(raw["counts"], dtype=float)  # (2, runs, steps_shuffled)
#     inv = np.empty_like(order)
#     inv[order] = np.arange(len(order))
#     counts_unshuf = counts[:, :, inv]

#     raw["counts"] = counts_unshuf.tolist()
#     raw.update(
#         {
#             "center_freq_ghz": float(center_freq_ghz),
#             "span_mhz": float(span_mhz),
#             "freqs_ghz": freqs_ghz.tolist(),
#             "freq_units": "GHz",
#             "mw_dur_ns": int(mw_dur_ns),
#             "uwave_ind": int(uwave_ind),
#             "uwave_power_dbm": uwave_power_dbm,
#             "shuffle_freqs": bool(shuffle_freqs),
#             "shuffle_seed": int(shuffle_seed),
#             "base_args": base_args,
#         }
#     )

#     if do_save:
#         try:
#             timestamp = dm.get_time_stamp()
#             nv_name = _get_nv_name(nv_sig)
#             file_path = dm.get_file_path(__file__, timestamp, nv_name)
#             raw["timestamp"] = timestamp
#             dm.save_raw_data(raw, file_path)
#             print(f"[ESR] saved: {file_path}")
#         except Exception:
#             print("[SAVE ERROR]\n", traceback.format_exc())

#     res = analyze_esr_2exp(
#         raw,
#         smooth_w=7,
#         do_plot=do_plot,
#         save_path=None,
#         title=f"ESR around {center_freq_ghz:.6f} GHz",
#     )

#     print("Fit success:", res["fit"]["success"])
#     print("f0 (GHz):", res["fit"]["f0_ghz"])
#     print("FWHM (MHz):", res["fit"]["fwhm_mhz"])
#     print("Contrast:", res["fit"]["contrast"])

#     tb.reset_cfm()
#     return raw, res
