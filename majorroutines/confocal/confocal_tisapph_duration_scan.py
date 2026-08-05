# -*- coding: utf-8 -*-
# Author: Yael Sternfeld
"""
Ti:sapph pulse-duration scan at a FIXED wavelength.

Companion routine to confocal_resonance_singlet_scan.py. Instead of sweeping
Ti:sapph wavelength at fixed probe duration, this sweeps probe_ns (the
Ti:sapph-on duration before/through the readout window) at a single fixed
wavelength. This directly tests whether the Ti:sapph-induced common-mode
brightening builds up with exposure duration (consistent with charge-state
population kinetics, e.g. NV- <-> NV0 cycling) or is flat/instantaneous
(consistent with an electronic/optical artifact rather than a population
process).

Assumes the same 4-gate sequence interface as the wavelength-scan routine:
    gate 0 = ms0, Ti:sapph OFF
    gate 1 = ms0, Ti:sapph ON
    gate 2 = ms1, Ti:sapph OFF
    gate 3 = ms1, Ti:sapph ON

IMPORTANT DIFFERENCE from the wavelength scan:
    In the wavelength scan, wavelength is set via tisapph.set_wavelength_nm(),
    a separate hardware call that does NOT require reloading the pulse
    sequence, so stream_load() is called once per run.

    Here, probe_ns is a SEQUENCE ARGUMENT (it defines the pulse pattern
    itself, exactly like tau_ns in a Rabi scan). This means stream_load()
    CANNOT be hoisted out of the duration loop -- the sequence must be
    reloaded at every duration step, since the pulse pattern itself changes
    with each probe_ns value. This mirrors the same constraint documented
    for confocal_rabi.py.
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
    ms0_response = _safe_ratio(
        np.asarray(ms0_on, dtype=float) - np.asarray(ms0_off, dtype=float),
        np.asarray(ms0_off, dtype=float),
    )
    ms1_response = _safe_ratio(
        np.asarray(ms1_on, dtype=float) - np.asarray(ms1_off, dtype=float),
        np.asarray(ms1_off, dtype=float),
    )
    delta_response = ms1_response - ms0_response
    return ms0_response, ms1_response, delta_response


def _compute_all_metrics(ms0_off, ms0_on, ms1_off, ms1_on):
    ms0_response, ms1_response, delta_response = _compute_tisapph_response(
        ms0_off, ms0_on, ms1_off, ms1_on
    )
    return {
        "ms0_response": ms0_response,
        "ms1_response": ms1_response,
        "delta_response": delta_response,
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


def _build_live_figure():
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_raw = axes[0]
    ax_resp = axes[1]

    ax_raw.set_title("Ti:sapph duration scan (fixed wavelength)")
    ax_raw.set_ylabel("Raw counts")
    ax_raw.set_xscale("log")

    ax_resp.set_ylabel("Ti:sapph-induced response")
    ax_resp.set_xlabel("Ti:sapph probe duration, probe_ns (ns)")
    ax_resp.set_xscale("log")
    ax_resp.axhline(0, color="0.6", lw=1, ls="--")

    (line_ms0_off,) = ax_raw.plot([], [], "o-", label="ms=0, OFF")
    (line_ms1_off,) = ax_raw.plot([], [], "o-", label="ms=±1, OFF")
    (line_ms0_on,) = ax_raw.plot([], [], "s--", label="ms=0, ON")
    (line_ms1_on,) = ax_raw.plot([], [], "s--", label="ms=±1, ON")

    (line_resp0,) = ax_resp.plot([], [], "o-", label="ms=0 response")
    (line_resp1,) = ax_resp.plot([], [], "o-", label="ms=±1 response")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    handles = {
        "line_ms0_off": line_ms0_off,
        "line_ms1_off": line_ms1_off,
        "line_ms0_on": line_ms0_on,
        "line_ms1_on": line_ms1_on,
        "line_resp0": line_resp0,
        "line_resp1": line_resp1,
    }

    return fig, axes, handles


def _update_live_figure(
    probe_durations_ns,
    axes,
    handles,
    ms0_off_mean,
    ms1_off_mean,
    ms0_on_mean,
    ms1_on_mean,
    resp0_mean,
    resp1_mean,
):
    handles["line_ms0_off"].set_data(probe_durations_ns, ms0_off_mean)
    handles["line_ms1_off"].set_data(probe_durations_ns, ms1_off_mean)
    handles["line_ms0_on"].set_data(probe_durations_ns, ms0_on_mean)
    handles["line_ms1_on"].set_data(probe_durations_ns, ms1_on_mean)

    handles["line_resp0"].set_data(probe_durations_ns, resp0_mean)
    handles["line_resp1"].set_data(probe_durations_ns, resp1_mean)

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    plt.pause(0.01)


def main(
    nv_sig,
    wavelength_nm,
    probe_durations_ns,
    num_reps=None,
    num_runs=None,
    uwave_ind=None,
    uwave_freq_ghz=None,
    uwave_power_dbm=None,
    readout_ns=None,
    pol_ns=None,
    laser_power=None,
    optimize_between_runs=True,
    do_plot=True,
    shuffle=True,
    settle_s=0.25,
):
    """
    Ti:sapph pulse-duration scan at a fixed wavelength.

    Parameters
    ----------
    nv_sig : NVSig
        Standard NV signature object.
    wavelength_nm : float
        Fixed Ti:sapph wavelength for the whole scan (e.g. your best 860-870
        nm S-pol candidate, or wherever you want to characterize the
        brightening's time dependence).
    probe_durations_ns : array-like of int
        The probe_ns values to sweep. Recommend a log-spaced array spanning
        from your shortest achievable AOM/pulse-streamer switching time up
        through your normal operating value (e.g. currently ~1e6 ns), e.g.:
            probe_durations_ns = np.logspace(2, 6, 13)  # 100 ns to 1 ms
    num_reps, num_runs : int
        Same meaning as in the wavelength-scan routine.
    uwave_ind, uwave_freq_ghz, uwave_power_dbm : optional
        Same meaning as in the wavelength-scan routine; pulled from the
        virtual sig-gen config if not given.
    readout_ns, pol_ns : optional
        Same meaning as in the wavelength-scan routine; pulled from
        nv_sig.pulse_durations / virtual laser config if not given.
    laser_power : optional
        Recorded in the output metadata only (not currently set by this
        routine -- add a call here if your Ti:sapph power needs explicit
        setting per run).
    optimize_between_runs, do_plot, shuffle, settle_s :
        Same meaning as in the wavelength-scan routine. `settle_s` here
        guards against any transient after reloading the sequence, not
        against a wavelength change (there is none -- wavelength is fixed).
    """
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

    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_freq_ghz is None:
        uwave_freq_ghz = float(vsg["frequency"])
    if uwave_power_dbm is None:
        uwave_power_dbm = float(vsg["uwave_power"])

    seq_file = "resonance_tisapph_singlet_scan.py"

    probe_durations_ns = np.asarray(probe_durations_ns, dtype=np.int64)
    num_steps = len(probe_durations_ns)

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

    print(f"seq file  = {seq_file}")
    print(f"wavelength= {wavelength_nm} nm (FIXED)")
    print(f"freq      = {uwave_freq_ghz} GHz")
    print(f"power     = {uwave_power_dbm} dBm")
    print(f"pol       = {pol_ns} ns")
    print(f"read      = {readout_ns} ns")
    print(f"probe_ns  = {probe_durations_ns.tolist()}")
    print(f"steps     = {num_steps}")
    print(f"reps      = {num_reps}")
    print(f"runs      = {num_runs}")

    tb.init_safe_stop()
    start_time = time.time()

    try:
        # Wavelength is fixed for the whole scan -- set it once up front.
        tisapph.set_wavelength_nm(float(wavelength_nm))
        time.sleep(settle_s)

        for run_ind in range(num_runs):
            print(f"\nRun {run_ind + 1}/{num_runs}")

            if tb.safe_stop():
                break

            if optimize_between_runs:
                targeting.compensate_for_drift(nv_sig, no_crash=True)
                # Re-set wavelength defensively in case drift compensation
                # touched anything shared with the Ti:sapph path.
                tisapph.set_wavelength_nm(float(wavelength_nm))
                time.sleep(settle_s)

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

                    probe_ns = int(probe_durations_ns[step_ind])

                    # probe_ns is a SEQUENCE ARGUMENT here (unlike wavelength),
                    # so the sequence must be reloaded at every step -- this
                    # cannot be hoisted out of the loop.
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

                    counter_server.clear_buffer()
                    pulsegen_server.stream_start(int(num_reps))

                    # read_counter_summed returns [gate0_total, gate1_total, ...]
                    # Gate order: ms0_off, ms0_on, ms1_off, ms1_on.
                    new_counts = counter_server.read_counter_summed(int(num_reps))
                    ms0_off_counts[run_ind, step_ind] = int(new_counts[0])
                    ms0_on_counts[run_ind, step_ind] = int(new_counts[1])
                    ms1_off_counts[run_ind, step_ind] = int(new_counts[2])
                    ms1_on_counts[run_ind, step_ind] = int(new_counts[3])

                    ms0_off_val = ms0_off_counts[run_ind, step_ind]
                    ms0_on_val = ms0_on_counts[run_ind, step_ind]
                    ms1_off_val = ms1_off_counts[run_ind, step_ind]
                    ms1_on_val = ms1_on_counts[run_ind, step_ind]

                    metrics_val = _compute_all_metrics(
                        ms0_off_val, ms0_on_val, ms1_off_val, ms1_on_val
                    )

                    print(
                        f"  probe_ns={probe_ns:>9d} | "
                        f"ms0_off={int(ms0_off_val)}, ms0_on={int(ms0_on_val)}, "
                        f"ms1_off={int(ms1_off_val)}, ms1_on={int(ms1_on_val)}, "
                        f"resp0={metrics_val['ms0_response']:.5e}, "
                        f"resp1={metrics_val['ms1_response']:.5e}"
                    )

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            # Per-run partial save, to avoid data loss on a long scan.
            _save_partial(
                nv_sig, timestamp, wavelength_nm, probe_durations_ns,
                num_reps, run_ind + 1, num_runs, uwave_ind, uwave_freq_ghz,
                uwave_power_dbm, pol_ns, readout_ns, laser_power,
                ms0_off_counts, ms0_on_counts, ms1_off_counts, ms1_on_counts,
                opti_coords_list,
            )

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
                resp0_mean, _ = _nanmean_ste(metrics_runs["ms0_response"])
                resp1_mean, _ = _nanmean_ste(metrics_runs["ms1_response"])

                _update_live_figure(
                    probe_durations_ns,
                    axes,
                    handles,
                    ms0_off_mean,
                    ms1_off_mean,
                    ms0_on_mean,
                    ms1_on_mean,
                    resp0_mean,
                    resp1_mean,
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

    raw_data = _package_raw_data(
        nv_sig, timestamp, elapsed_s, wavelength_nm, probe_durations_ns,
        num_reps, num_runs, uwave_ind, uwave_freq_ghz, uwave_power_dbm,
        pol_ns, readout_ns, laser_power,
        ms0_off_counts, ms0_on_counts, ms1_off_counts, ms1_on_counts,
        opti_coords_list,
    )

    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    print(f"\nSaved data to {file_path}")
    print(f"Elapsed time = {elapsed_s:.1f} s")

    return raw_data


def _package_raw_data(
    nv_sig, timestamp, elapsed_s, wavelength_nm, probe_durations_ns,
    num_reps, num_runs, uwave_ind, uwave_freq_ghz, uwave_power_dbm,
    pol_ns, readout_ns, laser_power,
    ms0_off_counts, ms0_on_counts, ms1_off_counts, ms1_on_counts,
    opti_coords_list,
):
    metrics_all = _compute_all_metrics(
        ms0_off_counts, ms0_on_counts, ms1_off_counts, ms1_on_counts
    )

    ms0_off_mean, ms0_off_ste = _nanmean_ste(ms0_off_counts)
    ms1_off_mean, ms1_off_ste = _nanmean_ste(ms1_off_counts)
    ms0_on_mean, ms0_on_ste = _nanmean_ste(ms0_on_counts)
    ms1_on_mean, ms1_on_ste = _nanmean_ste(ms1_on_counts)

    resp0_mean, resp0_ste = _nanmean_ste(metrics_all["ms0_response"])
    resp1_mean, resp1_ste = _nanmean_ste(metrics_all["ms1_response"])
    delta_mean, delta_ste = _nanmean_ste(metrics_all["delta_response"])

    return {
        "timestamp": timestamp,
        "elapsed_s": elapsed_s,
        "nv_sig": nv_sig,
        "wavelength_nm": float(wavelength_nm),
        "probe_durations_ns": np.asarray(probe_durations_ns).tolist(),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": float(uwave_freq_ghz),
        "uwave_power_dbm": float(uwave_power_dbm),
        "pol_ns": int(pol_ns),
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
        "ms0_response_mean": resp0_mean.tolist(),
        "ms0_response_ste": resp0_ste.tolist(),
        "ms1_response_mean": resp1_mean.tolist(),
        "ms1_response_ste": resp1_ste.tolist(),
        "delta_response_mean": delta_mean.tolist(),
        "delta_response_ste": delta_ste.tolist(),
        "opti_coords_list": opti_coords_list,
    }


def _save_partial(
    nv_sig, timestamp, wavelength_nm, probe_durations_ns,
    num_reps, runs_done, num_runs, uwave_ind, uwave_freq_ghz,
    uwave_power_dbm, pol_ns, readout_ns, laser_power,
    ms0_off_counts, ms0_on_counts, ms1_off_counts, ms1_on_counts,
    opti_coords_list,
):
    """
    Per-run partial save (overwrites the same file each time), so a crash
    mid-scan doesn't lose completed runs. Mirrors the established
    per-run-saving pattern used elsewhere in this repo.
    """
    try:
        raw_data_partial = _package_raw_data(
            nv_sig, timestamp, np.nan, wavelength_nm, probe_durations_ns,
            num_reps, num_runs, uwave_ind, uwave_freq_ghz, uwave_power_dbm,
            pol_ns, readout_ns, laser_power,
            ms0_off_counts, ms0_on_counts, ms1_off_counts, ms1_on_counts,
            opti_coords_list,
        )
        raw_data_partial["runs_completed"] = int(runs_done)
        file_path = dm.get_file_path(
            __file__, timestamp, getattr(nv_sig, "name", "nv") + f"-run{runs_done:03d}"
        )
        dm.save_raw_data(raw_data_partial, file_path)
    except Exception:
        print("Warning: per-run partial save failed:")
        print(traceback.format_exc())


def plot_raw_counts_vs_duration(raw_data):
    """
    Quick standalone plot: raw counts (all 4 gates) vs probe duration,
    log-x, for a saved/loaded raw_data dict from this routine. Useful as a
    sanity check before trusting the normalized response plot -- e.g. to
    confirm OFF-gate counts stay flat with duration (as expected, since the
    OFF gate shouldn't depend on probe_ns) while ON-gate counts show
    whatever duration-dependence is actually present.
    """
    probe_durations_ns = np.asarray(raw_data["probe_durations_ns"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(probe_durations_ns, raw_data["ms0_off_mean"], yerr=raw_data["ms0_off_ste"],
                fmt="o-", label="ms=0, OFF")
    ax.errorbar(probe_durations_ns, raw_data["ms1_off_mean"], yerr=raw_data["ms1_off_ste"],
                fmt="o-", label="ms=±1, OFF")
    ax.errorbar(probe_durations_ns, raw_data["ms0_on_mean"], yerr=raw_data["ms0_on_ste"],
                fmt="s--", label="ms=0, ON")
    ax.errorbar(probe_durations_ns, raw_data["ms1_on_mean"], yerr=raw_data["ms1_on_ste"],
                fmt="s--", label="ms=±1, ON")
    ax.set_xscale("log")
    ax.set_xlabel("Ti:sapph probe duration, probe_ns (ns)")
    ax.set_ylabel("Raw counts")
    ax.set_title(f"Raw counts vs duration @ {raw_data.get('wavelength_nm', '?')} nm")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_response_vs_duration(raw_data):
    """
    Quick standalone plot: Ti:sapph-induced response vs probe duration,
    log-x, for a saved/loaded raw_data dict from this routine.
    """
    probe_durations_ns = np.asarray(raw_data["probe_durations_ns"], dtype=float)
    resp0_mean = np.asarray(raw_data["ms0_response_mean"])
    resp0_ste = np.asarray(raw_data["ms0_response_ste"])
    resp1_mean = np.asarray(raw_data["ms1_response_mean"])
    resp1_ste = np.asarray(raw_data["ms1_response_ste"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(probe_durations_ns, resp0_mean, yerr=resp0_ste, fmt="o-", label="ms=0 response")
    ax.errorbar(probe_durations_ns, resp1_mean, yerr=resp1_ste, fmt="o-", label="ms=±1 response")
    ax.axhline(0, color="0.6", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Ti:sapph probe duration, probe_ns (ns)")
    ax.set_ylabel("Ti:sapph-induced response, (ON-OFF)/OFF")
    ax.set_title(f"Duration scan @ {raw_data.get('wavelength_nm', '?')} nm")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example usage -- edit before running:
    #
    # from utils.tool_belt import get_nv_sig  # or however nv_sig is loaded
    # nv_sig = get_nv_sig("Wu")
    #
    # probe_durations_ns = np.logspace(2, 6, 13).astype(np.int64)  # 100 ns .. 1 ms
    #
    # raw_data = main(
    #     nv_sig,
    #     wavelength_nm=866.0,       # pick your candidate wavelength
    #     probe_durations_ns=probe_durations_ns,
    #     num_reps=300_000,
    #     num_runs=5,
    #     uwave_ind=0,
    # )
    pass
