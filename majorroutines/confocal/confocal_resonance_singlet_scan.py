# -*- coding: utf-8 -*-
"""
Ti:sapph wavelength sweep for singlet / triplet-sensitive readout.

Assumes the sequence file returns 4 APD-gated experiments per repetition:
    gate 0 = ms0, Ti:sapph OFF
    gate 1 = ms0, Ti:sapph ON
    gate 2 = ms1, Ti:sapph OFF
    gate 3 = ms1, Ti:sapph ON

This updated version:
- keeps the same 4-gate sequence interface
- loads the sequence once per run (not once per wavelength)
- reapplies MW settings every run, like the working resonance driver
- adds live plots for:
    * raw counts, Ti:sapph OFF
    * raw counts, Ti:sapph ON
    * direct spin contrast, Ti:sapph OFF and ON
    * Ti:sapph-induced response for ms=0 and ms=±1
    * delta traces
"""

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey
import majorroutines.targeting as targeting


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)


def _compute_tisapph_response(ms0_off, ms0_on, ms1_off, ms1_on):
    """
    Ti:sapph-induced fractional response:
        ms0_response = (ms0_on - ms0_off) / ms0_off
        ms1_response = (ms1_on - ms1_off) / ms1_off
        delta_response = ms1_response - ms0_response
    """
    ms0_response = _safe_ratio(np.asarray(ms0_on, dtype=float) - np.asarray(ms0_off, dtype=float),
                               np.asarray(ms0_off, dtype=float))
    ms1_response = _safe_ratio(np.asarray(ms1_on, dtype=float) - np.asarray(ms1_off, dtype=float),
                               np.asarray(ms1_off, dtype=float))
    delta_response = ms1_response - ms0_response
    return ms0_response, ms1_response, delta_response


def _compute_spin_contrast(ms0, ms1):
    """
    Direct spin contrast:
        (ms0 - ms1) / ms0
    """
    return _safe_ratio(np.asarray(ms0, dtype=float) - np.asarray(ms1, dtype=float),
                       np.asarray(ms0, dtype=float))


def _compute_all_metrics(ms0_off, ms0_on, ms1_off, ms1_on):
    ms0_response, ms1_response, delta_response = _compute_tisapph_response(
        ms0_off, ms0_on, ms1_off, ms1_on
    )
    spin_contrast_off = _compute_spin_contrast(ms0_off, ms1_off)
    spin_contrast_on = _compute_spin_contrast(ms0_on, ms1_on)
    delta_spin_contrast = spin_contrast_on - spin_contrast_off

    return {
        "ms0_response": ms0_response,
        "ms1_response": ms1_response,
        "delta_response": delta_response,
        "spin_contrast_off": spin_contrast_off,
        "spin_contrast_on": spin_contrast_on,
        "delta_spin_contrast": delta_spin_contrast,
    }


def _nanmean_ste(arr):
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr, axis=0)

    counts = np.sum(np.isfinite(arr), axis=0)
    ste = np.full_like(mean, np.nan, dtype=float)

    valid = counts > 1
    if np.any(valid):
        std = np.nanstd(arr[:, valid], axis=0, ddof=1)
        ste[valid] = std / np.sqrt(counts[valid])

    return mean, ste


# def _build_live_figure():
#     fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

#     ax_raw_off = axes[0]
#     ax_raw_on = axes[1]
#     ax_spin = axes[2]
#     ax_resp = axes[3]
#     ax_delta = axes[4]

#     ax_raw_off.set_title("Ti:sapph singlet scan")
#     ax_raw_off.set_ylabel("Raw counts")
#     ax_raw_on.set_ylabel("Raw counts")
#     ax_spin.set_ylabel("Spin contrast")
#     ax_resp.set_ylabel("Ti:sapph response")
#     ax_delta.set_ylabel("Delta")
#     ax_delta.set_xlabel("Ti:sapph wavelength (nm)")

#     (line_ms0_off,) = ax_raw_off.plot([], [], "o-", label="ms=0, OFF")
#     (line_ms1_off,) = ax_raw_off.plot([], [], "o-", label="ms=±1, OFF")

#     (line_ms0_on,) = ax_raw_on.plot([], [], "o-", label="ms=0, ON")
#     (line_ms1_on,) = ax_raw_on.plot([], [], "o-", label="ms=±1, ON")

#     (line_spin_off,) = ax_spin.plot([], [], "o-", label="(ms0_off - ms1_off)/ms0_off")
#     (line_spin_on,) = ax_spin.plot([], [], "o-", label="(ms0_on - ms1_on)/ms0_on")

#     (line_resp_ms0,) = ax_resp.plot([], [], "o-", label="(ms0_on - ms0_off)/ms0_off")
#     (line_resp_ms1,) = ax_resp.plot([], [], "o-", label="(ms1_on - ms1_off)/ms1_off")

#     (line_delta_resp,) = ax_delta.plot([], [], "o-", label="resp(ms±1) - resp(ms0)")
#     (line_delta_spin,) = ax_delta.plot([], [], "o-", label="spin_on - spin_off")

#     for ax in axes:
#         ax.grid(True, linestyle="--", alpha=0.5)
#         ax.legend()

#     handles = {
#         "line_ms0_off": line_ms0_off,
#         "line_ms1_off": line_ms1_off,
#         "line_ms0_on": line_ms0_on,
#         "line_ms1_on": line_ms1_on,
#         "line_spin_off": line_spin_off,
#         "line_spin_on": line_spin_on,
#         "line_resp_ms0": line_resp_ms0,
#         "line_resp_ms1": line_resp_ms1,
#         "line_delta_resp": line_delta_resp,
#         "line_delta_spin": line_delta_spin,
#     }

#     return fig, axes, handles


# def _update_live_figure(
#     wavelengths_nm,
#     axes,
#     handles,
#     ms0_off_mean,
#     ms1_off_mean,
#     ms0_on_mean,
#     ms1_on_mean,
#     spin_off_mean,
#     spin_on_mean,
#     resp_ms0_mean,
#     resp_ms1_mean,
#     delta_resp_mean,
#     delta_spin_mean,
# ):
#     handles["line_ms0_off"].set_data(wavelengths_nm, ms0_off_mean)
#     handles["line_ms1_off"].set_data(wavelengths_nm, ms1_off_mean)

#     handles["line_ms0_on"].set_data(wavelengths_nm, ms0_on_mean)
#     handles["line_ms1_on"].set_data(wavelengths_nm, ms1_on_mean)

#     handles["line_spin_off"].set_data(wavelengths_nm, spin_off_mean)
#     handles["line_spin_on"].set_data(wavelengths_nm, spin_on_mean)

#     handles["line_resp_ms0"].set_data(wavelengths_nm, resp_ms0_mean)
#     handles["line_resp_ms1"].set_data(wavelengths_nm, resp_ms1_mean)

#     handles["line_delta_resp"].set_data(wavelengths_nm, delta_resp_mean)
#     handles["line_delta_spin"].set_data(wavelengths_nm, delta_spin_mean)

#     for ax in axes:
#         ax.relim()
#         ax.autoscale_view()

#     plt.pause(0.01)

def _build_live_figure():
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_raw_off = axes[0]
    ax_raw_on = axes[1]

    ax_raw_off.set_title("Ti:sapph singlet scan")
    ax_raw_off.set_ylabel("Raw counts")
    ax_raw_on.set_ylabel("Raw counts")
    ax_raw_on.set_xlabel("Ti:sapph wavelength (nm)")

    (line_ms0_off,) = ax_raw_off.plot([], [], "o-", label="ms=0, OFF")
    (line_ms1_off,) = ax_raw_off.plot([], [], "o-", label="ms=±1, OFF")

    (line_ms0_on,) = ax_raw_on.plot([], [], "o-", label="ms=0, ON")
    (line_ms1_on,) = ax_raw_on.plot([], [], "o-", label="ms=±1, ON")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    handles = {
        "line_ms0_off": line_ms0_off,
        "line_ms1_off": line_ms1_off,
        "line_ms0_on": line_ms0_on,
        "line_ms1_on": line_ms1_on,
    }

    return fig, axes, handles

def _update_live_figure(
    wavelengths_nm,
    axes,
    handles,
    ms0_off_mean,
    ms1_off_mean,
    ms0_on_mean,
    ms1_on_mean,
):
    handles["line_ms0_off"].set_data(wavelengths_nm, ms0_off_mean)
    handles["line_ms1_off"].set_data(wavelengths_nm, ms1_off_mean)

    handles["line_ms0_on"].set_data(wavelengths_nm, ms0_on_mean)
    handles["line_ms1_on"].set_data(wavelengths_nm, ms1_on_mean)

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    plt.pause(0.01)

def main(
    nv_sig,
    wavelength_start_nm=805.0,
    wavelength_stop_nm=825.0,
    num_steps=101,
    num_reps=1000,
    num_runs=10,
    uwave_ind=0,
    uwave_freq_ghz=None,
    uwave_power_dbm=None,
    probe_ns=5000,
    readout_ns=None,
    pol_ns=None,
    laser_power=None,
    optimize_between_runs=True,
    do_plot=True,
    shuffle=True,
    settle_s=0.25,
):
    tb.reset_cfm()
    kpl.init_kplotlib()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()
    tisapph = tb.get_server_tisapph()

    readout_vkey = VirtualLaserKey.SPIN_READOUT
    vld_read = tb.get_virtual_laser_dict(readout_vkey)
    if readout_ns is None:
        readout_ns = int(
            nv_sig.pulse_durations.get(readout_vkey, int(vld_read["duration"]))
        )
    readout_ns = int(readout_ns)

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    if pol_ns is None:
        pol_ns = int(
            nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_pol["duration"]))
        )
    pol_ns = int(pol_ns)

    probe_ns = int(probe_ns)

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_freq_ghz is None:
        uwave_freq_ghz = float(vsg["frequency"])
    if uwave_power_dbm is None:
        uwave_power_dbm = float(vsg["uwave_power"])

    seq_file = "resonance_tisapph_singlet_scan.py"
    seq_args = [
        int(pol_ns),
        int(probe_ns),
        int(readout_ns),
        int(uwave_ind),
        spin_pol_vkey,
        readout_vkey,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)

    wavelengths_nm = np.linspace(wavelength_start_nm, wavelength_stop_nm, num_steps)

    ms0_off_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms0_on_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms1_off_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms1_on_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()
    opti_coords_list = []

    if do_plot:
        fig, axes, handles = _build_live_figure()
    else:
        fig = None
        axes = None
        handles = None

    print(f"seq file = {seq_file}")
    print(f"seq args = {seq_args}")
    print(f"freq     = {uwave_freq_ghz} GHz")
    print(f"power    = {uwave_power_dbm} dBm")
    print(f"pol      = {pol_ns} ns")
    print(f"probe    = {probe_ns} ns")
    print(f"read     = {readout_ns} ns")
    print(f"steps    = {num_steps}")
    print(f"reps     = {num_reps}")
    print(f"runs     = {num_runs}")

    tb.init_safe_stop()
    start_time = time.time()

    try:
        for run_ind in range(num_runs):
            print(f"\nRun {run_ind + 1}/{num_runs}")

            if tb.safe_stop():
                break

            if optimize_between_runs:
                targeting.compensate_for_drift(nv_sig, no_crash=True)

            # Match the working resonance-style path:
            # load once per run, then reapply MW state each run
            pulsegen_server.stream_load(seq_file, seq_args_string)
            sig_gen.set_amp(float(uwave_power_dbm))
            sig_gen.set_freq(float(uwave_freq_ghz))
            sig_gen.uwave_on()

            sweep_order = np.arange(num_steps)
            if shuffle:
                np.random.shuffle(sweep_order)

            counter_server.start_tag_stream()
            try:
                for step_ind in sweep_order:
                    if tb.safe_stop():
                        break

                    wl_nm = float(wavelengths_nm[step_ind])
                    tisapph.set_wavelength_nm(wl_nm)
                    time.sleep(settle_s)

                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))

                    new_counts = counter_server.read_counter_modulo_gates(4, int(num_reps))
                    count_arr = np.array(new_counts, dtype=np.int64)

                    ms0_off_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                    ms0_on_counts[run_ind, step_ind] = count_arr[:, 1].sum()
                    ms1_off_counts[run_ind, step_ind] = count_arr[:, 2].sum()
                    ms1_on_counts[run_ind, step_ind] = count_arr[:, 3].sum()

                    ms0_off_val = ms0_off_counts[run_ind, step_ind]
                    ms0_on_val = ms0_on_counts[run_ind, step_ind]
                    ms1_off_val = ms1_off_counts[run_ind, step_ind]
                    ms1_on_val = ms1_on_counts[run_ind, step_ind]

                    metrics_val = _compute_all_metrics(
                        ms0_off_val,
                        ms0_on_val,
                        ms1_off_val,
                        ms1_on_val,
                    )

                    # print(
                    #     f"  wl={wl_nm:.3f} nm | "
                    #     f"ms0_off={int(ms0_off_val)}, ms0_on={int(ms0_on_val)}, "
                    #     f"ms1_off={int(ms1_off_val)}, ms1_on={int(ms1_on_val)}, "
                    #     f"spin_off={metrics_val['spin_contrast_off']:.5e}, "
                    #     f"spin_on={metrics_val['spin_contrast_on']:.5e}, "
                    #     f"resp0={metrics_val['ms0_response']:.5e}, "
                    #     f"resp1={metrics_val['ms1_response']:.5e}"
                    # )

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            if do_plot:
                ms0_off_mean, _ = _nanmean_ste(ms0_off_counts[: run_ind + 1])
                ms1_off_mean, _ = _nanmean_ste(ms1_off_counts[: run_ind + 1])
                ms0_on_mean, _ = _nanmean_ste(ms0_on_counts[: run_ind + 1])
                ms1_on_mean, _ = _nanmean_ste(ms1_on_counts[: run_ind + 1])

                metrics_runs = _compute_all_metrics(
                    ms0_off_counts[: run_ind + 1],
                    ms0_on_counts[: run_ind + 1],
                    ms1_off_counts[: run_ind + 1],
                    ms1_on_counts[: run_ind + 1],
                )

                spin_off_mean, _ = _nanmean_ste(metrics_runs["spin_contrast_off"])
                spin_on_mean, _ = _nanmean_ste(metrics_runs["spin_contrast_on"])
                resp_ms0_mean, _ = _nanmean_ste(metrics_runs["ms0_response"])
                resp_ms1_mean, _ = _nanmean_ste(metrics_runs["ms1_response"])
                delta_resp_mean, _ = _nanmean_ste(metrics_runs["delta_response"])
                delta_spin_mean, _ = _nanmean_ste(metrics_runs["delta_spin_contrast"])

                _update_live_figure(
                    wavelengths_nm,
                    axes,
                    handles,
                    ms0_off_mean,
                    ms1_off_mean,
                    ms0_on_mean,
                    ms1_on_mean,
                )

    except Exception:
        print(traceback.format_exc())
        raise

    finally:
        try:
            sig_gen.uwave_off()
        except Exception:
            pass
        tb.reset_cfm()

    elapsed_s = time.time() - start_time

    metrics_all = _compute_all_metrics(
        ms0_off_counts,
        ms0_on_counts,
        ms1_off_counts,
        ms1_on_counts,
    )

    ms0_off_mean, ms0_off_ste = _nanmean_ste(ms0_off_counts)
    ms1_off_mean, ms1_off_ste = _nanmean_ste(ms1_off_counts)
    ms0_on_mean, ms0_on_ste = _nanmean_ste(ms0_on_counts)
    ms1_on_mean, ms1_on_ste = _nanmean_ste(ms1_on_counts)

    resp_ms0_mean, resp_ms0_ste = _nanmean_ste(metrics_all["ms0_response"])
    resp_ms1_mean, resp_ms1_ste = _nanmean_ste(metrics_all["ms1_response"])
    delta_resp_mean, delta_resp_ste = _nanmean_ste(metrics_all["delta_response"])

    spin_off_mean, spin_off_ste = _nanmean_ste(metrics_all["spin_contrast_off"])
    spin_on_mean, spin_on_ste = _nanmean_ste(metrics_all["spin_contrast_on"])
    delta_spin_mean, delta_spin_ste = _nanmean_ste(metrics_all["delta_spin_contrast"])

    raw_data = {
        "timestamp": timestamp,
        "elapsed_s": elapsed_s,
        "nv_sig": nv_sig,
        "wavelengths_nm": wavelengths_nm.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": float(uwave_freq_ghz),
        "uwave_power_dbm": float(uwave_power_dbm),
        "pol_ns": int(pol_ns),
        "probe_ns": int(probe_ns),
        "readout_ns": int(readout_ns),
        "laser_power": laser_power,
        "ms0_off_counts": ms0_off_counts.tolist(),
        "ms0_on_counts": ms0_on_counts.tolist(),
        "ms1_off_counts": ms1_off_counts.tolist(),
        "ms1_on_counts": ms1_on_counts.tolist(),
        "ms0_off_mean": ms0_off_mean.tolist(),
        "ms0_off_ste": ms0_off_ste.tolist(),
        "ms1_off_mean": ms1_off_mean.tolist(),
        "ms1_off_ste": ms1_off_ste.tolist(),
        "ms0_on_mean": ms0_on_mean.tolist(),
        "ms0_on_ste": ms0_on_ste.tolist(),
        "ms1_on_mean": ms1_on_mean.tolist(),
        "ms1_on_ste": ms1_on_ste.tolist(),
        # Ti:sapph response metrics (kept compatible with old naming)
        "contrast_ms0_mean": resp_ms0_mean.tolist(),
        "contrast_ms0_ste": resp_ms0_ste.tolist(),
        "contrast_ms1_mean": resp_ms1_mean.tolist(),
        "contrast_ms1_ste": resp_ms1_ste.tolist(),
        "delta_contrast_mean": delta_resp_mean.tolist(),
        "delta_contrast_ste": delta_resp_ste.tolist(),
        # New direct spin-contrast metrics
        "spin_contrast_off_mean": spin_off_mean.tolist(),
        "spin_contrast_off_ste": spin_off_ste.tolist(),
        "spin_contrast_on_mean": spin_on_mean.tolist(),
        "spin_contrast_on_ste": spin_on_ste.tolist(),
        "delta_spin_contrast_mean": delta_spin_mean.tolist(),
        "delta_spin_contrast_ste": delta_spin_ste.tolist(),
        "opti_coords_list": opti_coords_list,
    }

    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    print(f"\nSaved data to {file_path}")
    print(f"Elapsed time = {elapsed_s:.1f} s")

    return raw_data


def plot_raw_counts_from_loaded(raw_data):
    wavelengths_nm = np.asarray(raw_data["wavelengths_nm"], dtype=float)

    ms0_off_mean = np.asarray(raw_data["ms0_off_mean"], dtype=float)
    ms1_off_mean = np.asarray(raw_data["ms1_off_mean"], dtype=float)
    ms0_on_mean = np.asarray(raw_data["ms0_on_mean"], dtype=float)
    ms1_on_mean = np.asarray(raw_data["ms1_on_mean"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax = axes[0]
    ax.plot(wavelengths_nm, ms0_off_mean, "o-", label="ms=0, Ti:sapph OFF")
    ax.plot(wavelengths_nm, ms1_off_mean, "o-", label="ms=±1, Ti:sapph OFF")
    ax.set_ylabel("Raw counts")
    ax.set_title("Ti:sapph OFF")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    ax = axes[1]
    ax.plot(wavelengths_nm, ms0_on_mean, "o-", label="ms=0, Ti:sapph ON")
    ax.plot(wavelengths_nm, ms1_on_mean, "o-", label="ms=±1, Ti:sapph ON")
    ax.set_xlabel("Ti:sapph wavelength (nm)")
    ax.set_ylabel("Raw counts")
    ax.set_title("Ti:sapph ON")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_spin_contrast_from_loaded(raw_data):
    wavelengths_nm = np.asarray(raw_data["wavelengths_nm"], dtype=float)

    spin_off_mean = np.asarray(raw_data["spin_contrast_off_mean"], dtype=float)
    spin_off_ste = np.asarray(raw_data["spin_contrast_off_ste"], dtype=float)

    spin_on_mean = np.asarray(raw_data["spin_contrast_on_mean"], dtype=float)
    spin_on_ste = np.asarray(raw_data["spin_contrast_on_ste"], dtype=float)

    delta_spin_mean = np.asarray(raw_data["delta_spin_contrast_mean"], dtype=float)
    delta_spin_ste = np.asarray(raw_data["delta_spin_contrast_ste"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax = axes[0]
    ax.errorbar(
        wavelengths_nm,
        spin_off_mean,
        yerr=spin_off_ste,
        fmt="o-",
        capsize=3,
        label="OFF: (ms0_off - ms1_off)/ms0_off",
    )
    ax.errorbar(
        wavelengths_nm,
        spin_on_mean,
        yerr=spin_on_ste,
        fmt="o-",
        capsize=3,
        label="ON: (ms0_on - ms1_on)/ms0_on",
    )
    ax.set_ylabel("Spin contrast")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    ax = axes[1]
    ax.errorbar(
        wavelengths_nm,
        delta_spin_mean,
        yerr=delta_spin_ste,
        fmt="o-",
        capsize=3,
        label="spin_on - spin_off",
    )
    ax.set_xlabel("Ti:sapph wavelength (nm)")
    ax.set_ylabel("Δ spin contrast")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_tisapph_response_from_loaded(raw_data):
    wavelengths_nm = np.asarray(raw_data["wavelengths_nm"], dtype=float)

    resp_ms0_mean = np.asarray(raw_data["contrast_ms0_mean"], dtype=float)
    resp_ms0_ste = np.asarray(raw_data["contrast_ms0_ste"], dtype=float)

    resp_ms1_mean = np.asarray(raw_data["contrast_ms1_mean"], dtype=float)
    resp_ms1_ste = np.asarray(raw_data["contrast_ms1_ste"], dtype=float)

    delta_resp_mean = np.asarray(raw_data["delta_contrast_mean"], dtype=float)
    delta_resp_ste = np.asarray(raw_data["delta_contrast_ste"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    axes[0].errorbar(wavelengths_nm, resp_ms0_mean, yerr=resp_ms0_ste, fmt="o-", capsize=3)
    axes[0].set_ylabel("(ms0_on - ms0_off)/ms0_off")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].errorbar(wavelengths_nm, resp_ms1_mean, yerr=resp_ms1_ste, fmt="o-", capsize=3)
    axes[1].set_ylabel("(ms1_on - ms1_off)/ms1_off")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].errorbar(wavelengths_nm, delta_resp_mean, yerr=delta_resp_ste, fmt="o-", capsize=3)
    axes[2].set_ylabel("resp(ms±1) - resp(ms0)")
    axes[2].set_xlabel("Ti:sapph wavelength (nm)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_loaded_summary(raw_data):
    wavelengths_nm = np.asarray(raw_data["wavelengths_nm"], dtype=float)

    ms0_off_mean = np.asarray(raw_data["ms0_off_mean"], dtype=float)
    ms1_off_mean = np.asarray(raw_data["ms1_off_mean"], dtype=float)
    ms0_on_mean = np.asarray(raw_data["ms0_on_mean"], dtype=float)
    ms1_on_mean = np.asarray(raw_data["ms1_on_mean"], dtype=float)

    spin_off_mean = np.asarray(raw_data["spin_contrast_off_mean"], dtype=float)
    spin_on_mean = np.asarray(raw_data["spin_contrast_on_mean"], dtype=float)

    resp_ms0_mean = np.asarray(raw_data["contrast_ms0_mean"], dtype=float)
    resp_ms1_mean = np.asarray(raw_data["contrast_ms1_mean"], dtype=float)

    delta_resp_mean = np.asarray(raw_data["delta_contrast_mean"], dtype=float)
    delta_spin_mean = np.asarray(raw_data["delta_spin_contrast_mean"], dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

    axes[0].plot(wavelengths_nm, ms0_off_mean, "o-", label="ms=0, OFF")
    axes[0].plot(wavelengths_nm, ms1_off_mean, "o-", label="ms=±1, OFF")
    axes[0].set_ylabel("Raw counts")
    axes[0].set_title("Ti:sapph singlet scan summary")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(wavelengths_nm, ms0_on_mean, "o-", label="ms=0, ON")
    axes[1].plot(wavelengths_nm, ms1_on_mean, "o-", label="ms=±1, ON")
    axes[1].set_ylabel("Raw counts")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].plot(wavelengths_nm, spin_off_mean, "o-", label="spin OFF")
    axes[2].plot(wavelengths_nm, spin_on_mean, "o-", label="spin ON")
    axes[2].set_ylabel("Spin contrast")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    axes[3].plot(wavelengths_nm, resp_ms0_mean, "o-", label="resp ms0")
    axes[3].plot(wavelengths_nm, resp_ms1_mean, "o-", label="resp ms±1")
    axes[3].set_ylabel("Ti:sapph response")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend()

    axes[4].plot(wavelengths_nm, delta_resp_mean, "o-", label="Δ response")
    axes[4].plot(wavelengths_nm, delta_spin_mean, "o-", label="Δ spin contrast")
    axes[4].set_ylabel("Delta")
    axes[4].set_xlabel("Ti:sapph wavelength (nm)")
    axes[4].grid(True, linestyle="--", alpha=0.5)
    axes[4].legend()

    plt.tight_layout()
    return fig


def _compute_ms_contrast(ms0, ms1):
    ms0 = np.asarray(ms0, dtype=float)
    ms1 = np.asarray(ms1, dtype=float)
    return _safe_ratio(ms0 - ms1, ms0)

def plot_ms_contrast_from_loaded(raw_data, use_tisapph_on=False):
    wavelengths_nm = np.asarray(raw_data["wavelengths_nm"], dtype=float)

    if use_tisapph_on:
        ms0 = np.asarray(raw_data["ms0_on_counts"], dtype=float)
        ms1 = np.asarray(raw_data["ms1_on_counts"], dtype=float)
        label = "(ms0_on - ms1_on) / ms0_on"
    else:
        ms0 = np.asarray(raw_data["ms0_off_counts"], dtype=float)
        ms1 = np.asarray(raw_data["ms1_off_counts"], dtype=float)
        label = "(ms0_off - ms1_off) / ms0_off"

    spin_contrast_runs = _compute_ms_contrast(ms0, ms1)
    spin_contrast_mean = np.nanmean(spin_contrast_runs, axis=0)
    spin_contrast_ste = np.nanstd(spin_contrast_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(spin_contrast_runs), axis=0)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        wavelengths_nm,
        spin_contrast_mean,
        yerr=spin_contrast_ste,
        fmt="o-",
        capsize=3,
    )
    ax.set_xlabel("Ti:sapph wavelength (nm)")
    ax.set_ylabel(label)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig

def plot_ms0_ms1_raw_from_loaded(raw_data):
    wavelengths_nm = np.asarray(raw_data["wavelengths_nm"], dtype=float)

    ms0_off = np.asarray(raw_data["ms0_off_counts"], dtype=float)
    ms1_off = np.asarray(raw_data["ms1_off_counts"], dtype=float)
    ms0_on = np.asarray(raw_data["ms0_on_counts"], dtype=float)
    ms1_on = np.asarray(raw_data["ms1_on_counts"], dtype=float)

    ms0_off_mean = np.nanmean(ms0_off, axis=0)
    ms1_off_mean = np.nanmean(ms1_off, axis=0)
    ms0_on_mean = np.nanmean(ms0_on, axis=0)
    ms1_on_mean = np.nanmean(ms1_on, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax = axes[0]
    ax.plot(wavelengths_nm, ms0_off_mean, "o-", label="ms=0, Ti:Sapph off")
    ax.plot(wavelengths_nm, ms1_off_mean, "o-", label="ms=±1, Ti:Sapph off")
    ax.set_ylabel("Raw Counts")
    ax.set_title("Ti:Sapph OFF")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    ax = axes[1]
    ax.plot(wavelengths_nm, ms0_on_mean, "o-", label="ms=0, Ti:Sapph on")
    ax.plot(wavelengths_nm, ms1_on_mean, "o-", label="ms=±1, Ti:Sapph on")
    ax.set_xlabel("Ti:sapph wavelength (nm)")
    ax.set_ylabel("Raw Counts")
    ax.set_title("Ti:Sapph ON")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    return fig

import copy
import numpy as np
from utils import data_manager as dm
from utils import kplotlib as kpl


def _mean_ste(arr):
    arr = np.asarray(arr, dtype=float)
    mean = np.nanmean(arr, axis=0)

    counts = np.sum(np.isfinite(arr), axis=0)
    ste = np.full_like(mean, np.nan, dtype=float)

    valid = counts > 1
    if np.any(valid):
        std = np.nanstd(arr[:, valid], axis=0, ddof=1)
        ste[valid] = std / np.sqrt(counts[valid])

    return mean, ste


def combine_tisapph_raw_data(data1, data2):
    wl1 = np.asarray(data1["wavelengths_nm"], dtype=float)
    wl2 = np.asarray(data2["wavelengths_nm"], dtype=float)

    if len(wl1) != len(wl2) or not np.allclose(wl1, wl2):
        raise ValueError("The two files do not have the same wavelength grid.")

    combined = copy.deepcopy(data1)

    ms0_off = np.vstack([
        np.asarray(data1["ms0_off_counts"], dtype=float),
        np.asarray(data2["ms0_off_counts"], dtype=float),
    ])
    ms0_on = np.vstack([
        np.asarray(data1["ms0_on_counts"], dtype=float),
        np.asarray(data2["ms0_on_counts"], dtype=float),
    ])
    ms1_off = np.vstack([
        np.asarray(data1["ms1_off_counts"], dtype=float),
        np.asarray(data2["ms1_off_counts"], dtype=float),
    ])
    ms1_on = np.vstack([
        np.asarray(data1["ms1_on_counts"], dtype=float),
        np.asarray(data2["ms1_on_counts"], dtype=float),
    ])

    combined["ms0_off_counts"] = ms0_off.tolist()
    combined["ms0_on_counts"] = ms0_on.tolist()
    combined["ms1_off_counts"] = ms1_off.tolist()
    combined["ms1_on_counts"] = ms1_on.tolist()

    ms0_off_mean, ms0_off_ste = _mean_ste(ms0_off)
    ms0_on_mean, ms0_on_ste = _mean_ste(ms0_on)
    ms1_off_mean, ms1_off_ste = _mean_ste(ms1_off)
    ms1_on_mean, ms1_on_ste = _mean_ste(ms1_on)

    combined["ms0_off_mean"] = ms0_off_mean.tolist()
    combined["ms0_off_ste"] = ms0_off_ste.tolist()
    combined["ms0_on_mean"] = ms0_on_mean.tolist()
    combined["ms0_on_ste"] = ms0_on_ste.tolist()
    combined["ms1_off_mean"] = ms1_off_mean.tolist()
    combined["ms1_off_ste"] = ms1_off_ste.tolist()
    combined["ms1_on_mean"] = ms1_on_mean.tolist()
    combined["ms1_on_ste"] = ms1_on_ste.tolist()

    combined["num_runs"] = int(
        np.asarray(data1["ms0_off_counts"]).shape[0]
        + np.asarray(data2["ms0_off_counts"]).shape[0]
    )

    return combined

if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example:
    # data = dm.get_raw_data(file_stem="2026_04_03-05_20_17-(lovelace)", load_npz=True)
    # data = dm.get_raw_data(file_stem="2026_04_04-09_42_57-(lovelace)", load_npz=True)
    # plot_ms_contrast_from_loaded(data, use_tisapph_on=True)

    data1 = dm.get_raw_data(file_stem="2026_04_04-17_48_29-(lovelace)", load_npz=True)
    data2 = dm.get_raw_data(file_stem="2026_04_04-16_53_01-(lovelace)", load_npz=True)

    combined_data = combine_tisapph_raw_data(data1, data2)

    plot_ms0_ms1_raw_from_loaded(combined_data)
    kpl.show(block=True)
    # Replace this with your actual nv_sig object
    # raise RuntimeError("Load or define nv_sig, then call main(...) manually.")