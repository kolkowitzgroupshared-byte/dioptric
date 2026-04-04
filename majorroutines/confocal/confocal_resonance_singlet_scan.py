# -*- coding: utf-8 -*-
"""
Ti:sapph wavelength sweep for singlet / triplet-sensitive readout.

Assumes the sequence file returns 4 APD-gated experiments per repetition:
    gate 0 = ms0, Ti:sapph OFF
    gate 1 = ms0, Ti:sapph ON
    gate 2 = ms1, Ti:sapph OFF
    gate 3 = ms1, Ti:sapph ON

The Ti:sapph wavelength is stepped by LabRAD.
The Ti:sapph AOM is pulsed by the Pulse Streamer sequence.

Very close in style to the working confocal resonance routine:
- stream_load once
- set wavelength each step
- stream_start
- read_counter_modulo_gates(4, num_reps)

Returns:
    raw_data
"""

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from utils import positioning as pos
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode
import majorroutines.targeting as targeting


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)

def _compute_contrasts(ms0_off, ms0_on, ms1_off, ms1_on):
    """
    Inputs can be scalars, 1D arrays, or 2D arrays.

    Returns
    -------
    norm_ms0_off, norm_ms0_on, norm_ms1_off, norm_ms1_on,
    contrast_ms0, contrast_ms1, delta_contrast
    """
    norm_ms0_off = np.asarray(ms0_off, dtype=float)
    norm_ms0_on = np.asarray(ms0_on, dtype=float)
    norm_ms1_off = np.asarray(ms1_off, dtype=float)
    norm_ms1_on = np.asarray(ms1_on, dtype=float)

    contrast_ms0 = _safe_ratio(norm_ms0_on - norm_ms0_off, norm_ms0_off)
    contrast_ms1 = _safe_ratio(norm_ms1_on - norm_ms1_off, norm_ms1_off)
    delta_contrast = contrast_ms1 - contrast_ms0

    return (
        norm_ms0_off,
        norm_ms0_on,
        norm_ms1_off,
        norm_ms1_on,
        contrast_ms0,
        contrast_ms1,
        delta_contrast,
    )


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
    shuffle=False,
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
        readout_ns = int(nv_sig.pulse_durations.get(readout_vkey, int(vld_read["duration"])))
    readout_ns = int(readout_ns)
    print(f"Readout duration (ns): {readout_ns}")

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    vld_pol = tb.get_virtual_laser_dict(spin_pol_vkey)
    if pol_ns is None:
        pol_ns = int(nv_sig.pulse_durations.get(spin_pol_vkey, int(vld_pol["duration"])))
    pol_ns = int(pol_ns)
    print(f"Polarization duration (ns): {pol_ns}")

    probe_ns = int(probe_ns)
    print(f"Ti:sapph probe duration (ns): {probe_ns}")

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_freq_ghz is None:
        uwave_freq_ghz = vsg.get("frequency", None)
    if uwave_power_dbm is None:
        uwave_power_dbm = vsg.get("uwave_power", None)

    if uwave_power_dbm is not None:
        sig_gen.set_amp(float(uwave_power_dbm))
    if uwave_freq_ghz is not None:
        sig_gen.set_freq(float(uwave_freq_ghz))
    sig_gen.uwave_on()

    seq_file = "resonance_tisapph_singlet_scan.py"
    seq_args = [
        int(pol_ns),
        int(probe_ns),
        int(readout_ns),
        int(uwave_ind),
        spin_pol_vkey,
        readout_vkey,
    ]

    wavelengths_nm = np.linspace(wavelength_start_nm, wavelength_stop_nm, num_steps)

    # Raw counts per run / step
    ms0_off_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms0_on_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms1_off_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms1_on_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()
    opti_coords_list = []

    if do_plot:
        fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

        ax0, ax1, ax2 = axes
        ax0.set_ylabel("ms0 contrast")
        ax1.set_ylabel("ms±1 contrast")
        ax2.set_ylabel("Δ contrast")
        ax2.set_xlabel("Ti:sapph wavelength (nm)")

        ax0.set_title("Ti:sapph singlet scan")

        (line_ms0,) = ax0.plot([], [], "o-", label="(ON-OFF)/OFF")
        (line_ms1,) = ax1.plot([], [], "o-", label="(ON-OFF)/OFF")
        (line_delta,) = ax2.plot([], [], "o-", label="ms±1 - ms0")

        ax0.grid(True, linestyle="--", alpha=0.5)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax0.legend()
        ax1.legend()
        ax2.legend()
    else:
        fig = None
        line_ms0 = None
        line_ms1 = None
        line_delta = None

    tb.init_safe_stop()

    try:
        for run_ind in range(num_runs):
            print(f"Run {run_ind + 1}/{num_runs}")

            if tb.safe_stop():
                break
            
            ###
            if optimize_between_runs:
                targeting.compensate_for_drift(nv_sig, no_crash=True)

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
                    seq_args_string = tb.encode_seq_args(seq_args)
                    pulsegen_server.stream_load(seq_file, seq_args_string)

                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))

                    # Each row should be [ms0_off, ms0_on, ms1_off, ms1_on]
                    new_counts = counter_server.read_counter_modulo_gates(4, int(num_reps))
                    count_arr = np.array(new_counts, dtype=np.int64)

                    print(f" count_arr shape: {count_arr.shape}")  # should be (num_reps, 4)

                    ms0_off_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                    ms0_on_counts[run_ind, step_ind] = count_arr[:, 1].sum()
                    ms1_off_counts[run_ind, step_ind] = count_arr[:, 2].sum()
                    ms1_on_counts[run_ind, step_ind] = count_arr[:, 3].sum()

                    ms0_off_val = ms0_off_counts[run_ind, step_ind]
                    ms0_on_val = ms0_on_counts[run_ind, step_ind]
                    ms1_off_val = ms1_off_counts[run_ind, step_ind]
                    ms1_on_val = ms1_on_counts[run_ind, step_ind]

                    (
                        _,
                        _,
                        _,
                        _,
                        contrast_ms0_val,
                        contrast_ms1_val,
                        delta_contrast_val,
                    ) = _compute_contrasts(
                        ms0_off_val,
                        ms0_on_val,
                        ms1_off_val,
                        ms1_on_val,
                    )

                    print(
                        f"  wl={wl_nm:.3f} nm | "
                        f"ms0_off={int(ms0_off_val)}, ms0_on={int(ms0_on_val)}, "
                        f"ms1_off={int(ms1_off_val)}, ms1_on={int(ms1_on_val)}, "
                        f"c0={contrast_ms0_val:.5e}, c1={contrast_ms1_val:.5e}, "
                        f"delta={delta_contrast_val:.5e}"
                    )

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            if do_plot:
                (
                    _,
                    _,
                    _,
                    _,
                    contrast_ms0_runs,
                    contrast_ms1_runs,
                    delta_contrast_runs,
                ) = _compute_contrasts(
                    ms0_off_counts[: run_ind + 1],
                    ms0_on_counts[: run_ind + 1],
                    ms1_off_counts[: run_ind + 1],
                    ms1_on_counts[: run_ind + 1],
                )

                contrast_ms0_mean = np.nanmean(contrast_ms0_runs, axis=0)
                contrast_ms1_mean = np.nanmean(contrast_ms1_runs, axis=0)
                delta_contrast_mean = np.nanmean(delta_contrast_runs, axis=0)

                line_ms0.set_data(wavelengths_nm, contrast_ms0_mean)
                line_ms1.set_data(wavelengths_nm, contrast_ms1_mean)
                line_delta.set_data(wavelengths_nm, delta_contrast_mean)

                for ax in axes:
                    ax.relim()
                    ax.autoscale_view()

                plt.pause(0.01)

            # if do_targeting:
            #     try:
            #         opti_coords = pos.set_xyz_on_nv(nv_sig)
            #         opti_coords_list.append(opti_coords)
            #     except Exception as e:
            #         print(f"Targeting failed on run {run_ind}: {e}")
            #         opti_coords_list.append(None)

    except Exception:
        print(traceback.format_exc())
        raise

    finally:
        try:
            sig_gen.uwave_off()
        except Exception:
            pass
        tb.reset_cfm()

    (
        norm_ms0_off,
        norm_ms0_on,
        norm_ms1_off,
        norm_ms1_on,
        contrast_ms0_runs,
        contrast_ms1_runs,
        delta_contrast_runs,
    ) = _compute_contrasts(
        ms0_off_counts,
        ms0_on_counts,
        ms1_off_counts,
        ms1_on_counts,
    )

    contrast_ms0_mean = np.nanmean(contrast_ms0_runs, axis=0)
    contrast_ms1_mean = np.nanmean(contrast_ms1_runs, axis=0)
    delta_contrast_mean = np.nanmean(delta_contrast_runs, axis=0)

    contrast_ms0_ste = np.nanstd(contrast_ms0_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(contrast_ms0_runs), axis=0)
    )
    contrast_ms1_ste = np.nanstd(contrast_ms1_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(contrast_ms1_runs), axis=0)
    )
    delta_contrast_ste = np.nanstd(delta_contrast_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(delta_contrast_runs), axis=0)
    )

    raw_data = {
        "timestamp": timestamp,
        "nv_sig": nv_sig,
        "wavelengths_nm": wavelengths_nm.tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": uwave_freq_ghz,
        "uwave_power_dbm": uwave_power_dbm,
        "pol_ns": int(pol_ns),
        "probe_ns": int(probe_ns),
        "readout_ns": int(readout_ns),
        "laser_power": laser_power,
        "ms0_off_counts": ms0_off_counts.tolist(),
        "ms0_on_counts": ms0_on_counts.tolist(),
        "ms1_off_counts": ms1_off_counts.tolist(),
        "ms1_on_counts": ms1_on_counts.tolist(),
        "contrast_ms0_mean": contrast_ms0_mean.tolist(),
        "contrast_ms0_ste": contrast_ms0_ste.tolist(),
        "contrast_ms1_mean": contrast_ms1_mean.tolist(),
        "contrast_ms1_ste": contrast_ms1_ste.tolist(),
        "delta_contrast_mean": delta_contrast_mean.tolist(),
        "delta_contrast_ste": delta_contrast_ste.tolist(),
        "opti_coords_list": opti_coords_list,
    }

    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    print(f"Saved data to {file_path}")
    return raw_data

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
if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example:
    data = dm.get_raw_data(file_stem="2026_04_03-05_20_17-(lovelace)", load_npz=True)
    # plot_ms_contrast_from_loaded(data, use_tisapph_on=True)
    plot_ms0_ms1_raw_from_loaded(data)
    kpl.show(block=True)
    # Replace this with your actual nv_sig object
    # raise RuntimeError("Load or define nv_sig, then call main(...) manually.")