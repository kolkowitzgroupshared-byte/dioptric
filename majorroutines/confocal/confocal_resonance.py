# -*- coding: utf-8 -*-
"""
Electron spin resonance routine. Scans the microwave frequency, taking counts
at each point.
"""


from random import shuffle

import labrad
import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
from majorroutines import pulsed_resonance
from utils import common
from utils import kplotlib as kpl
from utils import positioning as positioning
from utils import tool_belt as tb
from utils.constants import NormStyle, States
from utils.kplotlib import KplColors
import utils.data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils.constants import (
    CollectionMode,
    CoordsKey,
    CountFormat,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)

def main(
    nv_sig,
    freq_center,
    freq_range,
    num_steps,
    num_runs,
    uwave_list,
    state=States.LOW,
):
    ### Initial calculations and setup

    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()

    # Set up the laser

    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"])))
    readout_s = readout_ns / 1e9
    readout_laser = vld["physical_name"]
    # Since this is CW we need the imaging readout rather than the spin
    # readout typically used for state detection

    readout_power=None
    file_name = "resonance.py"
    seq_args = [readout_ns, state.value, readout_laser, readout_power]
    seq_args_string = tb.encode_seq_args(seq_args)
    # print(seq_args)
    # return

    # Calculate the frequencies we need to set
    half_freq_range = freq_range / 2
    freq_low = freq_center - half_freq_range
    freq_high = freq_center + half_freq_range
    freqs = np.linspace(freq_low, freq_high, num_steps)
    freq_ind_list = list(range(num_steps))
    freq_ind_master_list = []

    # Set up our data structure, an array of NaNs that we'll fill
    # incrementally. NaNs are ignored by matplotlib, which is why they're
    # useful for us here.
    # We define 2D arrays, with the horizontal dimension for the frequency and
    # the veritical dimension for the index of the run.
    ref_counts = np.empty([num_runs, num_steps])
    ref_counts[:] = np.nan
    sig_counts = np.copy(ref_counts)

    opti_coords_list = []

    ### Get the starting time of the function

    start_timestamp = dm.get_time_stamp()

    # Create raw data figure for incremental plotting
    raw_fig, ax_sig_ref, ax_norm = pulsed_resonance.create_raw_data_figure(
        freq_center, freq_range, num_steps
    )
    # Set up a run indicator for incremental plotting
    run_indicator_text = "Run #{}/{}"
    text = run_indicator_text.format(0, num_runs)
    run_indicator_obj = kpl.anchored_text(ax_norm, text, loc=kpl.Loc.UPPER_RIGHT)

    ### Collect the data

    # Start 'Press enter to stop...'
    tb.init_safe_stop()

    for run_ind in range(num_runs):
        print("Run index: {}".format(run_ind))

        # Break out of the while if the user says stop
        if tb.safe_stop():
            break

        # Optimize and save the coords we found/ drift tracking
        opti_coords = targeting.main_with_cxn(nv_sig)
        opti_coords_list.append(opti_coords)

        # Laser setup
        # tb.set_filter(nv_sig, laser_key)
        # laser_power = tb.set_laser_power(nv_sig, laser_key)
        # Start the laser now to get rid of transient effects
        # tb.laser_on(cxn, laser_name, laser_power)
        # sg = tb.get_server_sig_gen(int(uwave_ind))
        # sg_dict = tb.get_virtual_sig_gen_dict(int(uwave_ind))

        ###uwave setup
        sig_gen = tb.get_server_sig_gen(uwave_list[0])
        sig_gen.uwave_on()

        # Load the APD task with two samples for each frequency step
        pulsegen_server.stream_load(file_name, seq_args_string)
        counter_server.start_tag_stream()

        # Shuffle the list of frequency indices so that we step through
        # them randomly
        shuffle(freq_ind_list)
        freq_ind_master_list.append(freq_ind_list)

        # Take a sample and increment the frequency
        for step_ind in range(num_steps):
            # Break out of the while if the user says stop
            if tb.safe_stop():
                break

            freq_ind = freq_ind_list[step_ind]
            # print(freqs[freq_ind])
            sig_gen.set_freq(freqs[freq_ind])

            # Start the timing stream
            counter_server.clear_buffer()
            pulsegen_server.stream_start()

            # Read the counts using parity to distinguish signal vs ref
            new_counts = counter_server.read_counter_modulo_gates(2, 1)
            sample_counts = new_counts[0]

            cur_run_sig_counts_summed = sample_counts[1]
            cur_run_ref_counts_summed = sample_counts[0]

            sig_counts[run_ind, freq_ind] = cur_run_sig_counts_summed
            ref_counts[run_ind, freq_ind] = cur_run_ref_counts_summed
            # break
            # norm= sum(sig_gate_counts) / sum(ref_gate_counts)
            # print(norm)

        counter_server.stop_tag_stream()

        ### Incremental plotting

        # Update the run indicator
        text = run_indicator_text.format(run_ind + 1, num_runs)
        run_indicator_obj.txt.set_text(text)

        # Average the counts over the iterations
        inc_sig_counts = sig_counts[: run_ind + 1]
        inc_ref_counts = ref_counts[: run_ind + 1]
        ret_vals = tb.process_counts(
            inc_sig_counts, inc_ref_counts, 1, readout, norm_style
        )
        (
            sig_counts_avg_kcps,
            ref_counts_avg_kcps,
            norm_avg_sig,
            norm_avg_sig_ste,
        ) = ret_vals

        kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
        kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
        kpl.plot_line_update(ax_norm, y=norm_avg_sig)
        # Save the data we have incrementally for long measurements

        rawData = {
            "start_timestamp": start_timestamp,
            "nv_sig": nv_sig,
            # 'nv_sig-units': tb.get_nv_sig_units(),
            "opti_coords_list": opti_coords_list,
            "opti_coords_list-units": "V",
            "freq_center": freq_center,
            "freq_center-units": "GHz",
            "freq_range": freq_range,
            "freq_range-units": "GHz",
            "num_steps": num_steps,
            "num_runs": num_runs,
            "freq_ind_master_list": freq_ind_master_list,
            "readout": readout,
            "readout-units": "ns",
            "sig_counts": sig_counts.astype(int).tolist(),
            "sig_counts-units": "counts",
            "ref_counts": ref_counts.astype(int).tolist(),
            "ref_counts-units": "counts",
        }

        # This will continuously be the same file path so we will overwrite
        # the existing file with the latest version
        file_path = tb.get_file_path(
            __file__, start_timestamp, nv_sig["name"], "incremental"
        )
        tb.save_raw_data(rawData, file_path)

    ### Process and plot the data

    ret_vals = tb.process_counts(sig_counts, ref_counts, 1, readout, norm_style)
    (
        sig_counts_avg_kcps,
        ref_counts_avg_kcps,
        norm_avg_sig,
        norm_avg_sig_ste,
    ) = ret_vals

    # Raw data
    kpl.plot_line_update(ax_sig_ref, line_ind=0, y=sig_counts_avg_kcps)
    kpl.plot_line_update(ax_sig_ref, line_ind=1, y=ref_counts_avg_kcps)
    kpl.plot_line_update(ax_norm, y=norm_avg_sig)
    run_indicator_obj.remove()

    ### Clean up and save the data

    tb.reset_cfm()

    timestamp = tb.get_time_stamp()

    rawData = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        # 'nv_sig-units': tb.get_nv_sig_units(),
        "opti_coords_list": opti_coords_list,
        "opti_coords_list-units": "V",
        "freq_center": freq_center,
        "freq_center-units": "GHz",
        "freq_range": freq_range,
        "freq_range-units": "GHz",
        "num_steps": num_steps,
        "num_runs": num_runs,
        "freq_ind_master_list": freq_ind_master_list,
        "uwave_power-units": "dBm",
        "readout": readout,
        "readout-units": "ns",
        "sig_counts": sig_counts.astype(int).tolist(),
        "sig_counts-units": "counts",
        "ref_counts": ref_counts.astype(int).tolist(),
        "ref_counts-units": "counts",
        "norm_avg_sig": norm_avg_sig.astype(float).tolist(),
        "norm_avg_sig-units": "arb",
        #               'norm_avg_sig_ste': norm_avg_sig_ste.astype(float).tolist(),
        #               'norm_avg_sig_ste-units': 'arb',
    }

    name = nv_sig["name"]
    filePath = dm.get_file_path(__file__, timestamp, name)
    dm.save_figure(raw_fig, filePath)
    dm.save_raw_data(rawData, filePath)

    # Use the pulsed_resonance fitting functions
    fit_func = None
    if False:
        fit_func, popt, pcov = pulsed_resonance.fit_resonance(
            freq_range,
            freq_center,
            num_steps,
            norm_avg_sig,
            norm_avg_sig_ste,
            ref_counts,
        )

    fit_fig = None
    if (fit_func is not None) and (popt is not None):
        fit_fig = pulsed_resonance.create_fit_figure(
            freq_range, freq_center, num_steps, norm_avg_sig, fit_func, popt
        )
    filePath = tb.get_file_path(__file__, timestamp, name + "-fit")
    if fit_fig is not None:
        tb.save_figure(fit_fig, filePath)

    # if fit_func == pulsed_resonance.single_gaussian_dip:
    #     print('Single resonance at {:.4f} GHz'.format(popt[2]))
    #     print('\n')
    #     return popt[2], None
    # elif fit_func == pulsed_resonance.double_gaussian_dip:
    #     print('Resonances at {:.4f} GHz and {:.4f} GHz'.format(popt[2], popt[5]))
    #     print('Splitting of {:d} MHz'.format(int((popt[5] - popt[2]) * 1000)))
    #     print('\n')
    #     return popt[2], popt[5]
    # else:
    #     print('No resonances found')
    #     print('\n')
    return None, None


###

if __name__ == "__main__":
    file = "2022_12_06-15_24_46-johnson-search"
    file_path = "pc_carr/branch_master/resonance/2022_12/incremental"
    data = tb.get_raw_data(file, file_path)

    freq_center = data["freq_center"]
    freq_range = data["freq_range"]
    num_steps = data["num_steps"]
    num_runs = data["num_runs"]
    ref_counts = data["ref_counts"][0:1]
    sig_counts = data["sig_counts"][0:1]
    print(len(ref_counts))
    ret_vals = tb.process_counts(ref_counts, sig_counts)
    (
        avg_ref_counts,
        avg_sig_counts,
        norm_avg_sig,
        ste_ref_counts,
        ste_sig_counts,
        norm_avg_sig_ste,
    ) = ret_vals
    # norm_avg_sig_ste = None

    fit_func, popt, pcov = pulsed_resonance.fit_resonance(
        freq_center, freq_range, num_steps, norm_avg_sig, norm_avg_sig_ste
    )

    # fit_func, popt, pcov = fit_resonance(freq_range, freq_center, num_steps,
    #                                norm_avg_sig, ref_counts)

    pulsed_resonance.create_fit_figure(
        freq_center, freq_range, num_steps, norm_avg_sig, fit_func, popt
    )


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