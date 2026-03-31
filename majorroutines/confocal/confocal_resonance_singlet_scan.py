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

from majorroutines import targeting
from utils import positioning as pos
from utils import tool_belt as tb
from utils import kplotlib as kpl
from utils import data_manager as dm
from utils.constants import VirtualLaserKey, NormMode, CoordsKey


def _safe_ratio(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def _compute_contrasts(ms1_off, ms1_on):
# def _compute_contrasts(ms0_off, ms0_on, ms1_off, ms1_on):
    """
    Inputs can be scalars, 1D arrays, or 2D arrays.

    Returns
    -------
    norm_ms0_off, norm_ms0_on, norm_ms1_off, norm_ms1_on,
    contrast_ms0, contrast_ms1, delta_contrast
    """
    # norm_ms0_off = np.asarray(ms0_off, dtype=float)
    # norm_ms0_on = np.asarray(ms0_on, dtype=float)
    norm_ms1_off = np.asarray(ms1_off, dtype=float)
    norm_ms1_on = np.asarray(ms1_on, dtype=float)

    # contrast_ms0 = _safe_ratio(norm_ms0_on - norm_ms0_off, norm_ms0_off)
    contrast_ms1 = _safe_ratio(norm_ms1_on - norm_ms1_off, norm_ms1_off)
    # delta_contrast = contrast_ms1 - contrast_ms0

    return (
        # norm_ms0_off,
        # norm_ms0_on,
        norm_ms1_off,
        norm_ms1_on,
        # contrast_ms0,
        contrast_ms1,
        # delta_contrast,
    )


def main(
    nv_sig,
    wavelength_start_nm=805.0,
    wavelength_stop_nm=807.0,
    num_steps=3,
    num_reps=20e4,
    num_runs=10,
    uwave_ind=0,
    uwave_freq_ghz=None,
    uwave_power_dbm=None,
    probe_ns=100e3,
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
    seq_args_string = tb.encode_seq_args(seq_args)
    pulsegen_server.stream_load(seq_file, seq_args_string)

    wavelengths_nm = np.linspace(wavelength_start_nm, wavelength_stop_nm, num_steps)

    # Raw counts per run / step
    # ms0_off_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    # ms0_on_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms1_off_counts = np.full((num_runs, num_steps), np.nan, dtype=float)
    ms1_on_counts = np.full((num_runs, num_steps), np.nan, dtype=float)

    timestamp = dm.get_time_stamp()
    opti_coords_list = []

    if do_plot:
        fig, axes = plt.subplots(1, 1, figsize=(7, 9), sharex=True)

        # ax0, ax1, ax2 = axes
        ax1 = axes

        # ax0.set_ylabel("ms0 contrast")
        ax1.set_ylabel("ms±1 contrast")
        # ax2.set_ylabel("Δ contrast")
        ax1.set_xlabel("Ti:sapph wavelength (nm)")

        ax1.set_title("Ti:sapph singlet scan")

        # (line_ms0,) = ax0.plot([], [], "o-", label="(ON-OFF)/OFF")
        (line_ms1,) = ax1.plot([], [], "o-", label="(ON-OFF)/OFF")
        # (line_delta,) = ax2.plot([], [], "o-", label="ms±1 - ms0")

        # ax0.grid(True, linestyle="--", alpha=0.5)
        ax1.grid(True, linestyle="--", alpha=0.5)
        # ax2.grid(True, linestyle="--", alpha=0.5)

        # ax0.legend()
        ax1.legend()
        # ax2.legend()
    else:
        fig = None
        # line_ms0 = None
        line_ms1 = None
        # line_delta = None

    tb.init_safe_stop()

    try:
        for run_ind in range(num_runs):
            print(f"Run {run_ind + 1}/{num_runs}")

            if tb.safe_stop():
                break

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

                    # Each row should be [ms0_off, ms0_on, ms1_off, ms1_on]
                    new_counts = counter_server.read_counter_modulo_gates(4, int(num_reps))
                    count_arr = np.array(new_counts, dtype=np.int64)

                    print(f" count_arr shape: {count_arr.shape}")  # should be (num_reps, 4)

                    # ms0_off_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                    # ms0_on_counts[run_ind, step_ind] = count_arr[:, 1].sum()
                    ms1_off_counts[run_ind, step_ind] = count_arr[:, 0].sum()
                    ms1_on_counts[run_ind, step_ind] = count_arr[:, 1].sum()

                    # ms0_off_val = ms0_off_counts[run_ind, step_ind]
                    # ms0_on_val = ms0_on_counts[run_ind, step_ind]
                    ms1_off_val = ms1_off_counts[run_ind, step_ind]
                    ms1_on_val = ms1_on_counts[run_ind, step_ind]

                    (
                        _,
                        _,
                        contrast_ms1_val,
                    ) = _compute_contrasts(
                        ms1_off_val,
                        ms1_on_val,
                    )

                    print(
                        f"  wl={wl_nm:.3f} nm | "
                        f"ms1_off={int(ms1_off_val)}, ms1_on={int(ms1_on_val)}, "
                        f"c1={contrast_ms1_val:.5e}"
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
                    contrast_ms1_runs,
                ) = _compute_contrasts(
                    ms1_off_counts[: run_ind + 1],
                    ms1_on_counts[: run_ind + 1],
                )

                contrast_ms1_mean = np.nanmean(contrast_ms1_runs, axis=0)

                # line_ms0.set_data(wavelengths_nm, contrast_ms0_mean)
                line_ms1.set_data(wavelengths_nm, contrast_ms1_mean)
                # line_delta.set_data(wavelengths_nm, delta_contrast_mean)

                ax1.relim()
                ax1.autoscale_view()

                plt.pause(0.01)

            if optimize_between_runs:
                try:
                    z_coords, z_counts = targeting.optimize(nv_sig, coords_key=CoordsKey.Z)
                    galvo_key = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
                    xy_coords, xy_counts = targeting.optimize(nv_sig, coords_key=galvo_key)
                    print(f"  Optimized: Z={z_coords}, XY={xy_coords}, counts={xy_counts}")
                except Exception as e:
                    print(f"  Optimization failed on run {run_ind}: {e}")
                for f_num in plt.get_fignums():
                    if not do_plot or plt.figure(f_num) is not fig:
                        plt.close(f_num)

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
        norm_ms1_off,
        norm_ms1_on,
        contrast_ms1_runs,
    ) = _compute_contrasts(
        ms1_off_counts,
        ms1_on_counts,
    )

    # contrast_ms0_mean = np.nanmean(contrast_ms0_runs, axis=0)
    contrast_ms1_mean = np.nanmean(contrast_ms1_runs, axis=0)
    # delta_contrast_mean = np.nanmean(delta_contrast_runs, axis=0)

    # contrast_ms0_ste = np.nanstd(contrast_ms0_runs, axis=0, ddof=1) / np.sqrt(
        # np.sum(np.isfinite(contrast_ms0_runs), axis=0)
    # )
    contrast_ms1_ste = np.nanstd(contrast_ms1_runs, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(contrast_ms1_runs), axis=0)
    )
    # delta_contrast_ste = np.nanstd(delta_contrast_runs, axis=0, ddof=1) / np.sqrt(
    # #     # np.sum(np.isfinite(delta_contrast_runs), axis=0)
    # )

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
        # "ms0_off_counts": ms0_off_counts.tolist(),
        # "ms0_on_counts": ms0_on_counts.tolist(),
        "ms1_off_counts": ms1_off_counts.tolist(),
        "ms1_on_counts": ms1_on_counts.tolist(),
        # "contrast_ms0_mean": contrast_ms0_mean.tolist(),
        # "contrast_ms0_ste": contrast_ms0_ste.tolist(),
        "contrast_ms1_mean": contrast_ms1_mean.tolist(),
        "contrast_ms1_ste": contrast_ms1_ste.tolist(),
        # "delta_contrast_mean": delta_contrast_mean.tolist(),
        # "delta_contrast_ste": delta_contrast_ste.tolist(),
        "opti_coords_list": opti_coords_list,
    }

    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    print(f"Saved data to {file_path}")
    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example:
    # data = dm.get_raw_data(file_stem="some_previous_file", load_npz=True)
    # nv_sig = data["nv_sig"]

    # Replace this with your actual nv_sig object
    raise RuntimeError("Load or define nv_sig, then call main(...) manually.")