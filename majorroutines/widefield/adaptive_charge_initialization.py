# -*- coding: utf-8 -*-
"""
Compare standard conditional charge initialization vs DMD-assisted version.

Both modes use the exact same QUA sequence:
    charge_state_conditional_init.py

Modes:
    old:
        Original conditional init. No DMD manipulation.

    dmd_all_on:
        DMD is kept in pass-all / block-none state.
        Useful DMD-control baseline.

    dmd_block_confirmed:
        Same sequence and same base_routine method, but after each readout,
        NVs classified as NV- are blocked on DMD for future attempts.

This version saves:
    - raw counts
    - images if save_images=True
    - nv_list
    - thresholds
    - dmd feedback masks
    - summary plot
    - optional average image frames by rep
    - optional blinking GIF

Created: 2026-06-23
"""

import json
import time
import traceback
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from majorroutines.widefield import base_routine
from utils import widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey

# =============================================================================
# Basic helpers
# =============================================================================

def _get_thresholds(nv_list):
    thresholds = []
    for ind, nv in enumerate(nv_list):
        threshold = getattr(nv, "threshold", None)
        if threshold is None or not np.isfinite(threshold):
            raise ValueError(
                f"NV {ind} has invalid threshold: {threshold}. "
                "Assign nv.threshold before running."
            )
        thresholds.append(float(threshold))
    return np.asarray(thresholds, dtype=float)


def _append_to_file_path(file_path, suffix):
    file_path = Path(file_path)
    return file_path.with_name(f"{file_path.name}-{suffix}")

def _copy_nv_list_with_margin(nv_list, confirm_margin_counts=0.0):
    """
    Keep original nv_list untouched.
    """
    nv_run_list = []
    for nv in nv_list:
        nv_copy = copy.copy(nv)
        nv_copy.threshold = float(nv.threshold) + float(confirm_margin_counts)
        nv_run_list.append(nv_copy)
    return nv_run_list


def _prepare_dmd_indices(num_nvs, dmd_indices=None):
    """
    dmd_indices maps local nv_list index -> DMD loaded index.

    If using full nv_list and DMD index equals nv_list index:
        dmd_indices=None

    If using subset:
        nv_sub = [nv_list[i] for i in selected_inds]
        dmd_indices=selected_inds
    """
    if dmd_indices is None:
        dmd_indices = np.arange(num_nvs, dtype=int)
    else:
        dmd_indices = np.asarray(dmd_indices, dtype=int)

    if len(dmd_indices) != num_nvs:
        raise ValueError(
            f"dmd_indices length {len(dmd_indices)} does not match num_nvs {num_nvs}"
        )

    return dmd_indices


def _dmd_block_confirmed(
    confirmed_mask,
    dmd_indices,
    dmd=None,
    dmd_radius_px=8,
    dmd_plane=230,
    use_dmd=True,
    dmd_settle_s=0.001,
    verbose=False,
):
    """
    Uses your DMD server's robust convention:

        block_loaded_indices(indices)

    Server behavior:
        background = white / pass
        selected disks = black / block

    So active/unconfirmed NVs remain in the normal pass background.
    """
    confirmed_mask = np.asarray(confirmed_mask, dtype=bool)
    dmd_indices = np.asarray(dmd_indices, dtype=int)

    confirmed_dmd_indices = dmd_indices[confirmed_mask].astype(int).tolist()

    if verbose:
        print(
            f"DMD BLOCK confirmed mirrors: "
            f"{len(confirmed_dmd_indices)} / {len(dmd_indices)}"
        )

    if not use_dmd:
        return confirmed_dmd_indices

    if dmd is None:
        dmd = tb.get_server_dmd()

    dmd.block_loaded_indices(
        json.dumps(confirmed_dmd_indices),
        int(dmd_radius_px),
        int(dmd_plane),
    )

    time.sleep(dmd_settle_s)

    return confirmed_dmd_indices

def _dmd_gate_minimal_indices(
    confirmed_mask,
    dmd_indices,
    dmd=None,
    dmd_radius_px=8,
    dmd_plane=230,
    use_dmd=True,
    dmd_settle_s=0.001,
    verbose=False,
):
    """
    Fast hybrid DMD gating.

    Goal:
        confirmed NVs should be blocked
        active/unconfirmed NVs should be passed

    Two equivalent ways:

        A) white/pass background + black/block confirmed disks
           draw num_confirmed disks

        B) black/block background + white/pass active disks
           draw num_active disks

    This function chooses the smaller one.
    """

    confirmed_mask = np.asarray(confirmed_mask, dtype=bool)
    dmd_indices = np.asarray(dmd_indices, dtype=int)

    active_mask = ~confirmed_mask

    confirmed_dmd_indices = dmd_indices[confirmed_mask].astype(int).tolist()
    active_dmd_indices = dmd_indices[active_mask].astype(int).tolist()

    num_confirmed = len(confirmed_dmd_indices)
    num_active = len(active_dmd_indices)

    if not use_dmd:
        return {
            "method": "none",
            "num_confirmed": num_confirmed,
            "num_active": num_active,
            "num_drawn": 0,
        }

    if dmd is None:
        dmd = tb.get_server_dmd()

    if num_confirmed <= num_active:
        # White/pass background, black/block confirmed NVs.
        method = "block_confirmed"
        dmd.block_loaded_indices(
            json.dumps(confirmed_dmd_indices),
            int(dmd_radius_px),
            int(dmd_plane),
        )
        num_drawn = num_confirmed

    else:
        # Black/block background, white/pass active NVs.
        method = "pass_active"
        dmd.pass_loaded_indices(
            json.dumps(active_dmd_indices),
            int(dmd_radius_px),
            int(dmd_plane),
        )
        num_drawn = num_active

    time.sleep(dmd_settle_s)

    if verbose:
        print(
            f"DMD hybrid gate: method={method}, "
            f"confirmed={num_confirmed}, active={num_active}, drawn={num_drawn}"
        )

    return {
        "method": method,
        "num_confirmed": num_confirmed,
        "num_active": num_active,
        "num_drawn": num_drawn,
    }

def _dmd_pass_all_block_none(
    dmd=None,
    dmd_radius_px=8,
    dmd_plane=230,
    use_dmd=True,
    dmd_settle_s=0.001,
):
    """
    Pass background, block no loaded NV sites.

    With your server:
        block_loaded_indices([]) = white/pass background, no blocked NV disks.
    """
    if not use_dmd:
        return

    if dmd is None:
        dmd = tb.get_server_dmd()

    dmd.block_loaded_indices(
        json.dumps([]),
        int(dmd_radius_px),
        int(dmd_plane),
    )

    time.sleep(dmd_settle_s)


# =============================================================================
# Charge-prep functions
# =============================================================================
def make_dmd_block_confirmed_charge_prep_fn(
    num_nvs,
    dmd_indices=None,
    dmd_radius_px=8,
    dmd_plane=230,
    use_dmd=True,
    dmd_settle_s=0.001,
    verbose=True,
    profile_records=None,
):
    dmd_indices = _prepare_dmd_indices(num_nvs, dmd_indices)
    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd() if use_dmd else None

    confirmed_nvm = np.zeros(num_nvs, dtype=bool)
    last_confirmed_mask = {"mask": None}
    run_counter = {"run_ind": -1}

    def update_dmd_if_needed(confirmed_mask, force=False):
        confirmed_mask = np.asarray(confirmed_mask, dtype=bool)
        num_blocked = int(np.sum(confirmed_mask))

        if (
            not force
            and last_confirmed_mask["mask"] is not None
            and np.array_equal(confirmed_mask, last_confirmed_mask["mask"])
        ):
            return {
                "dmd_changed": False,
                "dmd_s": 0.0,
                "num_blocked": num_blocked,
            }

        t0 = time.perf_counter()

        _dmd_block_confirmed(
            confirmed_mask,
            dmd_indices=dmd_indices,
            dmd=dmd,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            use_dmd=use_dmd,
            dmd_settle_s=dmd_settle_s,
            verbose=verbose,
        )

        t1 = time.perf_counter()

        last_confirmed_mask["mask"] = confirmed_mask.copy()

        return {
            "dmd_changed": True,
            "dmd_s": float(t1 - t0),
            "num_blocked": num_blocked,
        }

    def charge_prep_fn(rep_ind, nv_list, initial_states_list=None):
        nonlocal confirmed_nvm

        t_total0 = time.perf_counter()

        # --------------------------------------------------------------
        # Start of each run
        # --------------------------------------------------------------
        if rep_ind == 0:
            run_counter["run_ind"] += 1
            confirmed_nvm[:] = False
            last_confirmed_mask["mask"] = None

            dmd_info = update_dmd_if_needed(confirmed_nvm, force=True)

            t_total1 = time.perf_counter()

            if profile_records is not None:
                profile_records.append(
                    {
                        "mode": "dmd_block_confirmed",
                        "run_ind": int(run_counter["run_ind"]),
                        "rep_ind": int(rep_ind),
                        "phase": "rep0_start",
                        "newly_confirmed": 0,
                        "confirmed": 0,
                        "active": int(num_nvs),
                        "classify_s": 0.0,
                        "dmd_s": dmd_info["dmd_s"],
                        "dmd_changed": bool(dmd_info["dmd_changed"]),
                        "opx_s": 0.0,
                        "total_callback_s": float(t_total1 - t_total0),
                    }
                )

            if verbose:
                print(
                    f"[DMD block confirmed] run {run_counter['run_ind']}, "
                    f"rep {rep_ind}: start, "
                    f"dmd={dmd_info['dmd_s']:.3f}s, "
                    f"total={t_total1 - t_total0:.3f}s"
                )

            return

        # --------------------------------------------------------------
        # Classify previous rep
        # --------------------------------------------------------------
        t_class0 = time.perf_counter()

        if initial_states_list is not None:
            states = np.asarray(initial_states_list, dtype=bool)
            active_prev = ~confirmed_nvm
            newly_confirmed = active_prev & states
            confirmed_nvm[newly_confirmed] = True
        else:
            newly_confirmed = np.zeros(num_nvs, dtype=bool)

        active_mask = ~confirmed_nvm

        t_class1 = time.perf_counter()

        # --------------------------------------------------------------
        # DMD update
        # --------------------------------------------------------------
        dmd_info = update_dmd_if_needed(confirmed_nvm, force=False)

        # --------------------------------------------------------------
        # OPX stream update
        # --------------------------------------------------------------
        t_opx0 = time.perf_counter()

        pulse_gen.insert_input_stream(
            "_cache_target_list",
            active_mask.astype(bool).tolist(),
        )

        t_opx1 = time.perf_counter()
        t_total1 = time.perf_counter()

        classify_s = float(t_class1 - t_class0)
        opx_s = float(t_opx1 - t_opx0)
        total_s = float(t_total1 - t_total0)

        if profile_records is not None:
            profile_records.append(
                {
                    "mode": "dmd_block_confirmed",
                    "run_ind": int(run_counter["run_ind"]),
                    "rep_ind": int(rep_ind),
                    "phase": "normal_feedback",
                    "newly_confirmed": int(np.sum(newly_confirmed)),
                    "confirmed": int(np.sum(confirmed_nvm)),
                    "active": int(np.sum(active_mask)),
                    "classify_s": classify_s,
                    "dmd_s": dmd_info["dmd_s"],
                    "dmd_changed": bool(dmd_info["dmd_changed"]),
                    "opx_s": opx_s,
                    "total_callback_s": total_s,
                }
            )

        if verbose:
            print(
                f"[DMD block confirmed] run {run_counter['run_ind']}, "
                f"rep {rep_ind}: "
                f"new={int(np.sum(newly_confirmed))}, "
                f"confirmed={int(np.sum(confirmed_nvm))}/{num_nvs}, "
                f"active={int(np.sum(active_mask))}/{num_nvs}, "
                f"classify={classify_s:.3f}s, "
                f"dmd={dmd_info['dmd_s']:.3f}s, "
                f"opx={opx_s:.3f}s, "
                f"total={total_s:.3f}s"
            )

    return charge_prep_fn

def make_dmd_all_on_charge_prep_fn(
    num_nvs,
    dmd_radius_px=8,
    dmd_plane=230,
    use_dmd=True,
    dmd_settle_s=0.001,
    verbose=True,
):
    """
    DMD-control baseline.

    DMD is always pass-all / block-none.
    OPX target list is exactly the old conditional method:
        target NVs that were not NV- in previous readout.
    """

    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd() if use_dmd else None
    run_counter = {"run_ind": -1}

    def charge_prep_fn(rep_ind, nv_list, initial_states_list=None):
        if rep_ind == 0:
            run_counter["run_ind"] += 1

            _dmd_pass_all_block_none(
                dmd=dmd,
                dmd_radius_px=dmd_radius_px,
                dmd_plane=dmd_plane,
                use_dmd=use_dmd,
                dmd_settle_s=dmd_settle_s,
            )

            if verbose:
                print(
                    f"[DMD all-on] run {run_counter['run_ind']}, "
                    f"rep {rep_ind}: pass all, no charge prep"
                )

            return

        if initial_states_list is not None:
            states = np.asarray(initial_states_list, dtype=bool)
            target_mask = ~states
        else:
            target_mask = np.ones(num_nvs, dtype=bool)

        pulse_gen.insert_input_stream(
            "_cache_target_list",
            target_mask.astype(bool).tolist(),
        )

        if verbose:
            print(
                f"[DMD all-on] run {run_counter['run_ind']}, rep {rep_ind}: "
                f"target={int(np.sum(target_mask))}/{num_nvs}"
            )

    return charge_prep_fn


def make_final_check_charge_prep_fn(
    base_charge_prep_fn,
    num_nvs,
    final_check_rep_ind,
    mode="old",
    dmd_radius_px=8,
    dmd_plane=230,
    dmd_settle_s=0.001,
    verbose=True,
    profile_records=None,
):
    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd() if mode.startswith("dmd") else None

    def wrapped_charge_prep_fn(rep_ind, nv_list, initial_states_list=None):
        if rep_ind == final_check_rep_ind:
            t_total0 = time.perf_counter()

            # DMD pass-all
            t_dmd0 = time.perf_counter()

            if mode.startswith("dmd"):
                _dmd_pass_all_block_none(
                    dmd=dmd,
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    use_dmd=True,
                    dmd_settle_s=dmd_settle_s,
                )

            t_dmd1 = time.perf_counter()

            # OPX target list all false
            t_opx0 = time.perf_counter()

            pulse_gen.insert_input_stream(
                "_cache_target_list",
                np.zeros(num_nvs, dtype=bool).tolist(),
            )

            t_opx1 = time.perf_counter()
            t_total1 = time.perf_counter()

            dmd_s = float(t_dmd1 - t_dmd0)
            opx_s = float(t_opx1 - t_opx0)
            total_s = float(t_total1 - t_total0)

            if profile_records is not None:
                profile_records.append(
                    {
                        "mode": mode,
                        "run_ind": None,
                        "rep_ind": int(rep_ind),
                        "phase": "final_check",
                        "newly_confirmed": 0,
                        "confirmed": None,
                        "active": 0,
                        "classify_s": 0.0,
                        "dmd_s": dmd_s,
                        "dmd_changed": True,
                        "opx_s": opx_s,
                        "total_callback_s": total_s,
                    }
                )

            if verbose:
                print(
                    f"[final check] rep {rep_ind}: "
                    f"dmd_pass_all={dmd_s:.3f}s, "
                    f"opx_all_false={opx_s:.3f}s, "
                    f"total={total_s:.3f}s"
                )

            return

        return base_charge_prep_fn(rep_ind, nv_list, initial_states_list)

    return wrapped_charge_prep_fn

def make_timed_charge_prep_fn(
    base_charge_prep_fn,
    timing_records,
    verbose=False,
):
    """
    Record time at every rep.

    Saves:
        run_ind
        rep_ind
        elapsed_s_from_run_start
        wall_s
    """

    run_counter = {"run_ind": -1}
    run_t0 = {"t": None}

    def timed_charge_prep_fn(rep_ind, nv_list, initial_states_list=None):
        now = time.perf_counter()

        if rep_ind == 0:
            run_counter["run_ind"] += 1
            run_t0["t"] = now

        run_ind = run_counter["run_ind"]

        if run_t0["t"] is None:
            elapsed_s = np.nan
        else:
            elapsed_s = now - run_t0["t"]

        timing_records.append(
            {
                "run_ind": int(run_ind),
                "rep_ind": int(rep_ind),
                "elapsed_s_from_run_start": float(elapsed_s),
                "wall_s": float(now),
            }
        )

        if verbose:
            print(
                f"[timing] run {run_ind}, rep {rep_ind}, "
                f"elapsed {elapsed_s:.3f} s"
            )

        return base_charge_prep_fn(rep_ind, nv_list, initial_states_list)

    return timed_charge_prep_fn
# =============================================================================
# Reconstruction and plotting
# =============================================================================

def reconstruct_confirmed_history(raw_data, mode="old"):
    """
    Reconstruct state history from saved counts.

    raw_data["counts"] shape:
        [exp, nv, run, step, rep]

    For old / dmd_all_on:
        state is simply thresholded state each rep.

    For dmd_block_confirmed:
        confirmed state is persistent once true.
    """

    nv_list = raw_data["nv_list"]
    thresholds = _get_thresholds(nv_list)

    counts = np.asarray(raw_data["counts"], dtype=float)[0]  # [nv, run, step, rep]
    counts = counts[:, :, 0, :]                              # [nv, run, rep]

    num_nvs, num_runs, num_reps_total = counts.shape

    num_reps_analysis = int(
        raw_data.get(
            "num_reps_analysis",
            raw_data.get("num_reps_normal", num_reps_total),
        )
    )

    counts = counts[:, :, :num_reps_analysis]
    
    num_nvs, num_runs, num_reps = counts.shape
    
    raw_states = counts > thresholds[:, None, None]

    if mode in ["old", "dmd_all_on"]:
        confirmed_history = raw_states.copy()
        newly_confirmed_history = raw_states.copy()
        active_history = ~raw_states.copy()

    elif mode == "dmd_block_confirmed":
        confirmed_history = np.zeros((num_nvs, num_runs, num_reps), dtype=bool)
        newly_confirmed_history = np.zeros((num_nvs, num_runs, num_reps), dtype=bool)
        active_history = np.zeros((num_nvs, num_runs, num_reps), dtype=bool)

        for run_ind in range(num_runs):
            confirmed = np.zeros(num_nvs, dtype=bool)

            for rep_ind in range(num_reps):
                active_before = ~confirmed
                active_history[:, run_ind, rep_ind] = active_before

                newly = active_before & raw_states[:, run_ind, rep_ind]
                newly_confirmed_history[:, run_ind, rep_ind] = newly

                confirmed = confirmed | newly
                confirmed_history[:, run_ind, rep_ind] = confirmed

    else:
        raise ValueError(f"Unknown mode: {mode}")

    fraction_by_run = np.mean(confirmed_history, axis=0)  # [run, rep]

    return {
        "thresholds": thresholds,
        "counts_for_states": counts,
        "raw_states": raw_states,
        "confirmed_history": confirmed_history,
        "newly_confirmed_history": newly_confirmed_history,
        "active_history": active_history,
        "fraction_confirmed_by_run": fraction_by_run,
        "avg_fraction": np.mean(fraction_by_run, axis=0),
        "ste_fraction": (
            np.std(fraction_by_run, axis=0, ddof=1) / np.sqrt(num_runs)
            if num_runs > 1
            else np.zeros(num_reps)
        ),
    }


def process_and_plot(raw_data, mode=None, mean_val=None, save_fig=True):
    if mode is None:
        mode = raw_data.get("mode", "old")

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    num_runs = int(raw_data["num_runs"])
    num_reps_total = int(raw_data["num_reps"])
    num_reps = int(raw_data.get("num_reps_analysis", num_reps_total))
    
    feedback = reconstruct_confirmed_history(raw_data, mode=mode)
    raw_data["feedback_summary"] = feedback

    avg_fraction = feedback["avg_fraction"]
    ste_fraction = feedback["ste_fraction"]

    reps_vals = np.arange(num_reps)

    xlim = min(11, num_reps, len(avg_fraction))
    reps_vals_plot = reps_vals[:xlim]
    avg_fraction_plot = avg_fraction[:xlim]
    ste_fraction_plot = ste_fraction[:xlim]

    fig, ax = plt.subplots(figsize=kpl.figsize)

    kpl.plot_points(
        ax,
        reps_vals_plot,
        avg_fraction_plot,
        yerr=ste_fraction_plot,
    )

    ax.set_xlabel("Attempt index")
    ax.set_ylabel("Fraction of NVs in NV$^{-}$")    
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(MaxNLocator(integer=True))

    if mean_val is not None:
        ax.axhline(
            mean_val,
            color=kpl.KplColors.GREEN,
            linestyle="dashed",
            linewidth=2,
        )

    txt = (
        f"num NVs = {num_nvs}\n"
        f"num runs = {num_runs}\n"
        f"final NV$^-$ fraction = {avg_fraction[-1]:.3f}"
    )
    kpl.anchored_text(ax, txt, kpl.Loc.LOWER_RIGHT, size=kpl.Size.SMALL)

    # Optional fit
    try:
        good = np.isfinite(ste_fraction_plot) & (ste_fraction_plot > 0)
        if num_runs > 1 and np.sum(good) >= 4:

            def fit_fn(x, y0, c1, c2):
                term1 = (c1 ** x) * y0
                term2 = c2 * (1 - (c1 ** x)) / (1 - c1)
                return term1 + term2

            popt, pcov = curve_fit(
                fit_fn,
                reps_vals_plot[good],
                avg_fraction_plot[good],
                p0=(0.05, 0.7, 0.1),
                sigma=ste_fraction_plot[good],
                absolute_sigma=True,
                maxfev=10000,
            )

            reps_dense = np.linspace(0, xlim - 1, 1000)
            kpl.plot_line(ax, reps_dense, fit_fn(reps_dense, *popt))

            print("Fit popt:", popt)
            print("Fit perr:", np.sqrt(np.diag(pcov)))

    except Exception:
        print("Fit failed:")
        print(traceback.format_exc())
    
    
    if save_fig:
        timestamp = raw_data.get("timestamp", dm.get_time_stamp())
        file_path = dm.get_file_path(
            __file__,
            timestamp,
            f"plot-{mode}",
        )

        dm.save_figure(fig, file_path)
        print("Saved summary plot:", file_path)
        
    return fig


def plot_final_check_image(
    raw_data,
    mode=None,
    img_coords=None,
    clim=None,
    marker_size=16,
):
    """
    Plot the final-check image averaged over runs.

    This uses the extra final rep:
        final_check_rep_ind = raw_data["final_check_rep_ind"]

    The image is taken with DMD pass-all and no extra charge prep.
    """

    if "img_arrays" not in raw_data:
        print("No img_arrays in raw_data. Cannot plot final-check image.")
        return None

    if "final_check_rep_ind" not in raw_data:
        print("No final_check_rep_ind in raw_data. Skipping final-check plot.")
        return None

    final_rep_ind = int(raw_data["final_check_rep_ind"])

    img_arrays = np.asarray(raw_data["img_arrays"], dtype=float)

    # Expected shape: [exp, run, step, rep, y, x]
    imgs = img_arrays[0, :, 0, final_rep_ind, :, :]  # [run, y, x]
    avg_final_img = np.nanmean(imgs, axis=0)

    if clim is None:
        vmin = np.nanpercentile(avg_final_img, 50)
        vmax = np.nanpercentile(avg_final_img, 99.8)
        clim = (vmin, vmax)

    nv_list = raw_data["nv_list"]
    coords_xy = None

    try:
        coords_xy = _coerce_img_coords(nv_list, img_coords=img_coords)
    except Exception:
        coords_xy = None
        print("Could not overlay NV coordinates on final-check image.")

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        avg_final_img,
        vmin=clim[0],
        vmax=clim[1],
        origin="upper",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("photons")

    ax.set_title(
        f"Final-check image, rep {final_rep_ind}\n"
        "DMD pass-all, no extra charge prep",
        fontsize=14,
    )

    if coords_xy is not None:
        x = coords_xy[:, 0]
        y = coords_xy[:, 1]

        ax.scatter(
            x,
            y,
            s=marker_size,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
            label="previous NV positions",
        )

        ax.legend(loc="upper right", fontsize=8)

    ax.set_axis_off()

    return fig

def analyze_and_plot_final_check(
    raw_data,
    mode=None,
    min_loss_runs=None,
    borderline_window_counts=5.0,
    save_data=True,
    save_fig=True,
):
    """
    Analyze the independent final-check readout after adaptive initialization.

    An NV is counted as lost in a run when:
        previously confirmed NV- and final-check state is NV0.

    The figure contains only the per-run transition summary.
    """

    # ------------------------------------------------------------------
    # Basic data
    # ------------------------------------------------------------------

    mode = mode or raw_data.get("mode", "dmd_block_confirmed")

    if "final_check_rep_ind" not in raw_data:
        raise ValueError("raw_data does not contain final_check_rep_ind.")

    final_rep_ind = int(raw_data["final_check_rep_ind"])
    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)

    counts = np.asarray(raw_data["counts"], dtype=float)

    if counts.ndim != 5:
        raise ValueError(
            "Expected counts[exp, nv, run, step, rep]; "
            f"received shape {counts.shape}."
        )

    if counts.shape[1] != num_nvs:
        raise ValueError(
            f"counts has {counts.shape[1]} NVs, but nv_list has {num_nvs}."
        )

    if not 0 <= final_rep_ind < counts.shape[-1]:
        raise IndexError(
            f"final_check_rep_ind={final_rep_ind} is outside "
            f"0–{counts.shape[-1] - 1}."
        )

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    confirm_margin = float(
        raw_data.get("confirm_margin_counts", 0.0)
    )

    if "analysis_thresholds" in raw_data:
        regular_thresholds = np.asarray(
            raw_data["analysis_thresholds"],
            dtype=float,
        )
    elif "thresholds" in raw_data:
        regular_thresholds = (
            np.asarray(raw_data["thresholds"], dtype=float)
            - confirm_margin
        )
    else:
        regular_thresholds = (
            np.asarray(_get_thresholds(nv_list), dtype=float)
            - confirm_margin
        )

    if regular_thresholds.shape != (num_nvs,):
        raise ValueError(
            f"Expected {num_nvs} regular thresholds; "
            f"received shape {regular_thresholds.shape}."
        )

    # ------------------------------------------------------------------
    # Previous and final charge states
    # ------------------------------------------------------------------

    # final_counts[nv, run]
    final_counts = counts[
        0,
        :,
        :,
        0,
        final_rep_ind,
    ]

    num_runs = final_counts.shape[1]

    if min_loss_runs is None:
        min_loss_runs = num_runs // 2 + 1

    min_loss_runs = int(min_loss_runs)

    if not 1 <= min_loss_runs <= num_runs:
        raise ValueError(
            f"min_loss_runs must be between 1 and {num_runs}."
        )

    threshold_2d = regular_thresholds[:, None]
    final_delta = final_counts - threshold_2d
    final_nvm_mask = final_counts > threshold_2d

    feedback = reconstruct_confirmed_history(
        raw_data,
        mode=mode,
    )

    previous_nvm_mask = np.asarray(
        feedback["confirmed_history"],
        dtype=bool,
    )[:, :, -1]

    if previous_nvm_mask.shape != final_nvm_mask.shape:
        raise ValueError(
            "Previous and final charge-state shapes differ: "
            f"{previous_nvm_mask.shape} versus {final_nvm_mask.shape}."
        )

    feedback_thresholds = np.asarray(
        feedback["thresholds"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Transition masks
    # ------------------------------------------------------------------

    retained_mask = previous_nvm_mask & final_nvm_mask
    lost_mask = previous_nvm_mask & ~final_nvm_mask
    gained_mask = ~previous_nvm_mask & final_nvm_mask
    unconfirmed_mask = ~previous_nvm_mask & ~final_nvm_mask

    borderline_window_counts = abs(
        float(borderline_window_counts)
    )

    borderline_lost_mask = (
        lost_mask
        & (final_delta >= -borderline_window_counts)
    )

    clear_lost_mask = (
        lost_mask
        & ~borderline_lost_mask
    )

    # ------------------------------------------------------------------
    # Per-run statistics
    # ------------------------------------------------------------------

    previous_nvm_by_run = np.sum(
        previous_nvm_mask,
        axis=0,
    )
    retained_nvm_by_run = np.sum(
        retained_mask,
        axis=0,
    )
    lost_nvm_by_run = np.sum(
        lost_mask,
        axis=0,
    )
    clear_lost_by_run = np.sum(
        clear_lost_mask,
        axis=0,
    )
    borderline_lost_by_run = np.sum(
        borderline_lost_mask,
        axis=0,
    )
    gained_nvm_by_run = np.sum(
        gained_mask,
        axis=0,
    )
    final_nvm_by_run = np.sum(
        final_nvm_mask,
        axis=0,
    )
    unconfirmed_by_run = np.sum(
        unconfirmed_mask,
        axis=0,
    )

    retention_fraction_by_run = np.full(
        num_runs,
        np.nan,
        dtype=float,
    )
    loss_fraction_by_run = np.full(
        num_runs,
        np.nan,
        dtype=float,
    )

    valid_runs = previous_nvm_by_run > 0

    retention_fraction_by_run[valid_runs] = (
        retained_nvm_by_run[valid_runs]
        / previous_nvm_by_run[valid_runs]
    )

    loss_fraction_by_run[valid_runs] = (
        lost_nvm_by_run[valid_runs]
        / previous_nvm_by_run[valid_runs]
    )

    # ------------------------------------------------------------------
    # Per-NV repeated behavior
    # ------------------------------------------------------------------

    confirmed_count_by_nv = np.sum(
        previous_nvm_mask,
        axis=1,
    )
    retained_count_by_nv = np.sum(
        retained_mask,
        axis=1,
    )
    loss_count_by_nv = np.sum(
        lost_mask,
        axis=1,
    )
    clear_loss_count_by_nv = np.sum(
        clear_lost_mask,
        axis=1,
    )
    borderline_loss_count_by_nv = np.sum(
        borderline_lost_mask,
        axis=1,
    )

    repeatedly_lost_mask = (
        loss_count_by_nv >= min_loss_runs
    )
    repeatedly_clear_lost_mask = (
        clear_loss_count_by_nv >= min_loss_runs
    )
    repeatedly_borderline_mask = (
        repeatedly_lost_mask
        & ~repeatedly_clear_lost_mask
    )

    repeatedly_lost_inds = np.where(
        repeatedly_lost_mask
    )[0].astype(int).tolist()

    repeatedly_clear_lost_inds = np.where(
        repeatedly_clear_lost_mask
    )[0].astype(int).tolist()

    repeatedly_borderline_inds = np.where(
        repeatedly_borderline_mask
    )[0].astype(int).tolist()

    lost_all_runs_inds = np.where(
        loss_count_by_nv == num_runs
    )[0].astype(int).tolist()

    loss_probability_by_nv = np.full(
        num_nvs,
        np.nan,
        dtype=float,
    )

    eligible_nv_mask = confirmed_count_by_nv > 0

    loss_probability_by_nv[eligible_nv_mask] = (
        loss_count_by_nv[eligible_nv_mask]
        / confirmed_count_by_nv[eligible_nv_mask]
    )

    # ------------------------------------------------------------------
    # Mean statistics
    # ------------------------------------------------------------------

    mean_previous_nvm = float(
        np.mean(previous_nvm_by_run)
    )
    mean_retained_nvm = float(
        np.mean(retained_nvm_by_run)
    )
    mean_lost_nvm = float(
        np.mean(lost_nvm_by_run)
    )
    mean_clear_lost = float(
        np.mean(clear_lost_by_run)
    )
    mean_borderline_lost = float(
        np.mean(borderline_lost_by_run)
    )
    mean_gained_nvm = float(
        np.mean(gained_nvm_by_run)
    )
    mean_final_nvm = float(
        np.mean(final_nvm_by_run)
    )
    mean_retention_fraction = float(
        np.nanmean(retention_fraction_by_run)
    )
    mean_loss_fraction = float(
        np.nanmean(loss_fraction_by_run)
    )

    # ------------------------------------------------------------------
    # Saved summary
    # ------------------------------------------------------------------

    summary = {
        "analysis_type": "adaptive_final_check_transition_analysis",
        "mode": mode,
        "num_nvs": int(num_nvs),
        "num_runs": int(num_runs),
        "final_check_rep_ind": int(final_rep_ind),
        "confirm_margin_counts": confirm_margin,
        "min_loss_runs": min_loss_runs,
        "borderline_window_counts": borderline_window_counts,

        "regular_thresholds": regular_thresholds.tolist(),
        "feedback_thresholds": feedback_thresholds.tolist(),
        "final_counts": final_counts.tolist(),
        "final_delta_from_threshold": final_delta.tolist(),

        "previous_confirmed_mask": previous_nvm_mask.tolist(),
        "final_nvm_mask": final_nvm_mask.tolist(),
        "retained_mask": retained_mask.tolist(),
        "lost_mask": lost_mask.tolist(),
        "clear_lost_mask": clear_lost_mask.tolist(),
        "borderline_lost_mask": borderline_lost_mask.tolist(),
        "gained_mask": gained_mask.tolist(),

        "previous_nvm_by_run": previous_nvm_by_run.tolist(),
        "retained_nvm_by_run": retained_nvm_by_run.tolist(),
        "lost_nvm_by_run": lost_nvm_by_run.tolist(),
        "clear_lost_by_run": clear_lost_by_run.tolist(),
        "borderline_lost_by_run": borderline_lost_by_run.tolist(),
        "gained_nvm_by_run": gained_nvm_by_run.tolist(),
        "final_nvm_by_run": final_nvm_by_run.tolist(),
        "unconfirmed_by_run": unconfirmed_by_run.tolist(),
        "retention_fraction_by_run": (
            retention_fraction_by_run.tolist()
        ),
        "loss_fraction_by_run": (
            loss_fraction_by_run.tolist()
        ),

        "confirmed_count_by_nv": confirmed_count_by_nv.tolist(),
        "retained_count_by_nv": retained_count_by_nv.tolist(),
        "loss_count_by_nv": loss_count_by_nv.tolist(),
        "clear_loss_count_by_nv": clear_loss_count_by_nv.tolist(),
        "borderline_loss_count_by_nv": (
            borderline_loss_count_by_nv.tolist()
        ),
        "loss_probability_by_nv": (
            loss_probability_by_nv.tolist()
        ),

        "repeatedly_lost_inds": repeatedly_lost_inds,
        "repeatedly_clear_lost_inds": (
            repeatedly_clear_lost_inds
        ),
        "repeatedly_borderline_inds": (
            repeatedly_borderline_inds
        ),
        "lost_all_runs_inds": lost_all_runs_inds,

        "mean_previous_nvm": mean_previous_nvm,
        "mean_retained_nvm": mean_retained_nvm,
        "mean_lost_nvm": mean_lost_nvm,
        "mean_clear_lost": mean_clear_lost,
        "mean_borderline_lost": mean_borderline_lost,
        "mean_gained_nvm": mean_gained_nvm,
        "mean_final_nvm": mean_final_nvm,
        "mean_retention_fraction": (
            mean_retention_fraction
        ),
        "mean_loss_fraction": mean_loss_fraction,
    }

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    print(
        "\n=== Adaptive final-check transition analysis ==="
    )
    print("mode:", mode)
    print("NVs:", num_nvs)
    print("runs:", num_runs)
    print("final-check rep:", final_rep_ind)
    print(
        "repeated-loss criterion:",
        f"{min_loss_runs}/{num_runs} runs",
    )

    for run_ind in range(num_runs):
        print(
            f"Run {run_ind}: "
            f"previous={previous_nvm_by_run[run_ind]}, "
            f"retained={retained_nvm_by_run[run_ind]}, "
            f"lost={lost_nvm_by_run[run_ind]}, "
            f"clear={clear_lost_by_run[run_ind]}, "
            f"borderline={borderline_lost_by_run[run_ind]}, "
            f"gained={gained_nvm_by_run[run_ind]}, "
            f"final={final_nvm_by_run[run_ind]}"
        )

    print("\nMean previous NV-:", mean_previous_nvm)
    print("Mean retained NV-:", mean_retained_nvm)
    print("Mean lost NV- -> NV0:", mean_lost_nvm)
    print("Mean final NV-:", mean_final_nvm)
    print("Mean retention:", mean_retention_fraction)
    print("Repeatedly lost:", repeatedly_lost_inds)
    print("Lost in every run:", lost_all_runs_inds)

    # ------------------------------------------------------------------
    # Save analysis
    # ------------------------------------------------------------------

    timestamp = raw_data.get(
        "timestamp",
        dm.get_time_stamp(),
    )

    if save_data:
        save_path = dm.get_file_path(
            __file__,
            timestamp,
            f"final-check-transition-summary-{mode}",
        )

        dm.save_raw_data(
            summary,
            save_path,
        )

        print(
            "Saved final-check transition summary:",
            save_path,
        )

    # ------------------------------------------------------------------
    # Plot only the per-run summary
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5),
    )

    run_inds = np.arange(num_runs)

    ax.bar(
        run_inds,
        retained_nvm_by_run,
        color=kpl.KplColors.GREEN,
        alpha=0.55,
        label="Retained previous NV$^-$",
    )

    ax.bar(
        run_inds,
        lost_nvm_by_run,
        bottom=retained_nvm_by_run,
        color=kpl.KplColors.RED,
        alpha=0.55,
        label="Lost NV$^- \\rightarrow$ NV$^0$",
    )

    ax.plot(
        run_inds,
        final_nvm_by_run,
        "o--",
        color=kpl.KplColors.BLUE,
        linewidth=1.7,
        markersize=5,
        label="Total final NV$^-$",
    )

    ax.set_title(
        "Per-run final-check transitions",
        fontsize=14,
    )
    ax.set_xlabel(
        "Run index",
        fontsize=13,
    )
    ax.set_ylabel(
        "Number of NVs",
        fontsize=13,
    )
    ax.set_xticks(run_inds)
    ax.set_ylim(
        0,
        num_nvs * 1.05,
    )
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )
    ax.legend(
        fontsize=9,
        loc="lower left",
    )
    ax.tick_params(
        labelsize=11,
    )

    summary_text = (
        f"Mean previous NV$^-$ = {mean_previous_nvm:.1f}\n"
        f"Mean final NV$^-$ = {mean_final_nvm:.1f}\n"
        f"Mean retention = {mean_retention_fraction:.3f}\n"
        f"Mean lost/run = {mean_lost_nvm:.1f}\n"
        # f"Clear loss/run = {mean_clear_lost:.1f}\n"
        # f"Borderline loss/run = {mean_borderline_lost:.1f}\n"
        # f"Repeatedly lost = {len(repeatedly_lost_inds)}"
    )

    kpl.anchored_text(
        ax,
        summary_text,
        kpl.Loc.LOWER_RIGHT,
        size=kpl.Size.SMALL,
    )

    fig.suptitle(
        "Adaptive initialization: independent final verification",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94],
    )

    if save_fig:
        fig_path = dm.get_file_path(
            __file__,
            timestamp,
            f"final-check-transition-plot-{mode}",
        )

        dm.save_figure(
            fig,
            fig_path,
        )

        print(
            "Saved final-check transition plot:",
            fig_path,
        )

    raw_data[
        "final_check_charge_summary"
    ] = summary

    return summary, fig

def _rep_timing_array(raw_data):
    records = raw_data.get("rep_timing_records", None)

    if records is None or len(records) == 0:
        return None

    num_runs = int(raw_data["num_runs"])
    num_reps_total = int(
        raw_data.get(
            "num_reps_total",
            raw_data.get("num_reps", 0),
        )
    )

    timing = np.full((num_runs, num_reps_total), np.nan, dtype=float)

    for rec in records:
        run_ind = int(rec["run_ind"])
        rep_ind = int(rec["rep_ind"])

        if 0 <= run_ind < num_runs and 0 <= rep_ind < num_reps_total:
            timing[run_ind, rep_ind] = float(rec["elapsed_s_from_run_start"])

    return timing


def plot_rep_timing_summary(raw_data):
    """
    Plot timing averaged across runs.

    Top:
        elapsed time from start of run

    Bottom:
        time between consecutive reps
    """

    timing = _rep_timing_array(raw_data)

    if timing is None:
        print("No rep_timing_records found. Skipping timing plot.")
        return None

    num_runs, num_reps_total = timing.shape
    reps = np.arange(num_reps_total)

    mean_elapsed = np.nanmean(timing, axis=0)
    ste_elapsed = (
        np.nanstd(timing, axis=0, ddof=1) / np.sqrt(num_runs)
        if num_runs > 1
        else np.zeros(num_reps_total)
    )

    delta = np.full_like(timing, np.nan)
    delta[:, 0] = 0.0
    delta[:, 1:] = np.diff(timing, axis=1)

    mean_delta = np.nanmean(delta, axis=0)
    ste_delta = (
        np.nanstd(delta, axis=0, ddof=1) / np.sqrt(num_runs)
        if num_runs > 1
        else np.zeros(num_reps_total)
    )

    final_check_rep_ind = raw_data.get("final_check_rep_ind", None)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].errorbar(
        reps,
        mean_elapsed,
        yerr=ste_elapsed,
        marker="o",
        linestyle="-",
        capsize=3,
    )
    axes[0].set_ylabel("Elapsed time from run start (s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        reps,
        mean_delta,
        yerr=ste_delta,
        marker="o",
        linestyle="-",
        capsize=3,
    )
    axes[1].set_xlabel("Rep index")
    axes[1].set_ylabel("Time since previous rep (s)")
    axes[1].grid(True, alpha=0.3)

    if final_check_rep_ind is not None:
        final_check_rep_ind = int(final_check_rep_ind)

        for ax in axes:
            ax.axvline(
                final_check_rep_ind,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="final check",
            )
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Rep timing averaged over {num_runs} runs",
        fontsize=14,
    )

    plt.tight_layout()
    return fig
# =============================================================================
# Image saving helpers
# =============================================================================

def save_avg_rep_images(raw_data, file_path, reps_to_save=None, clim=None):
    """
    Save average image for selected reps.

    Requires raw_data["img_arrays"], from base_routine with:
        save_images=True
        save_images_avg_reps=False

    img_arrays shape:
        [exp, run, step, rep, y, x]
    """

    if "img_arrays" not in raw_data:
        print("No img_arrays in raw_data. Skipping image frame saving.")
        return []

    img_arrays = np.asarray(raw_data["img_arrays"], dtype=float)

    # [run, rep, y, x]
    imgs = img_arrays[0, :, 0, :, :, :]

    num_runs, num_reps = imgs.shape[:2]

    if reps_to_save is None:
        reps_to_save = list(range(min(num_reps, 10)))

    saved_figs = []

    for rep_ind in reps_to_save:
        if rep_ind < 0 or rep_ind >= num_reps:
            continue

        avg_img = np.nanmean(imgs[:, rep_ind], axis=0)

        fig, ax = plt.subplots()
        kpl.imshow(
            ax,
            avg_img,
            title=f"Average image, rep {rep_ind}",
            cbar_label="photons",
            clim=clim,
        )
        ax.set_axis_off()

        rep_file_path = _append_to_file_path(file_path, f"avg-img-rep{rep_ind}")
        dm.save_figure(fig, rep_file_path)

        saved_figs.append(fig)

    return saved_figs


def save_blink_gif(
    raw_data,
    file_path,
    max_reps=20,
    clim=None,
    interval_ms=250,
):
    """
    Save clean blink GIF over rep/attempt index.

    No cumulative logic.
    No artificial bright patches.
    Just raw camera frames averaged over runs.

    Requires:
        raw_data["img_arrays"]

    img_arrays expected shape:
        [exp, run, step, rep, y, x]
    """

    if "img_arrays" not in raw_data:
        print("No img_arrays in raw_data. Skipping GIF.")
        return None

    file_path = Path(file_path)

    img_arrays = np.asarray(raw_data["img_arrays"], dtype=float)

    # [run, rep, y, x]
    imgs = img_arrays[0, :, 0, :, :, :]

    num_runs, num_reps = imgs.shape[:2]
    num_frames = min(num_reps, max_reps)

    # Average over runs for each rep.
    avg_imgs = np.nanmean(imgs[:, :num_frames], axis=0)  # [rep, y, x]

    if clim is None:
        vmin = np.nanpercentile(avg_imgs, 80)
        vmax = np.nanpercentile(avg_imgs, 99.8)
    else:
        vmin, vmax = clim

    fig, ax = plt.subplots()
    ax.set_axis_off()

    im = ax.imshow(avg_imgs[0], vmin=vmin, vmax=vmax)
    title = ax.set_title(f"rep 0, avg over {num_runs} runs")

    def update(rep_ind):
        im.set_data(avg_imgs[rep_ind])
        title.set_text(f"rep {rep_ind}, avg over {num_runs} runs")
        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval_ms,
        blit=False,
    )

    stem = file_path.stem if file_path.suffix else file_path.name
    gif_path = file_path.with_name(f"{stem}-blink-avg-over-runs.gif")

    fps = max(1, int(1000 / interval_ms))
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))

    plt.close(fig)

    print("Saved raw-frame blink GIF:", gif_path)
    return gif_path


from scipy.ndimage import gaussian_filter

def save_blink_gif(
    raw_data,
    file_path,
    max_reps=20,
    clim=None,
    interval_ms=250,
    background_percentile=5,
    bg_smooth_sigma=2,
    frame_smooth_sigma=0,
    contrast_percentiles=(50, 99.8),
    subtract_background=True,
):
    """
    Save clean blink GIF over rep/attempt index.

    No cumulative logic.
    No artificial bright patches.

    Movie frame:
        average over runs at each rep
        optionally subtract smooth low-percentile background

    img_arrays expected shape:
        [exp, run, step, rep, y, x]
    """

    if "img_arrays" not in raw_data:
        print("No img_arrays in raw_data. Skipping GIF.")
        return None

    file_path = Path(file_path)

    img_arrays = np.asarray(raw_data["img_arrays"], dtype=float)

    # [run, rep, y, x]
    imgs = img_arrays[0, :, 0, :, :, :]

    num_runs, num_reps = imgs.shape[:2]
    num_frames = min(num_reps, max_reps)

    # Average over runs for each rep.
    avg_imgs = np.nanmean(imgs[:, :num_frames], axis=0)  # [rep, y, x]

    if subtract_background:
        # Pixel-wise low-percentile background across reps.
        bg_img = np.nanpercentile(
            avg_imgs,
            background_percentile,
            axis=0,
        )

        if bg_smooth_sigma is not None and bg_smooth_sigma > 0:
            bg_img = gaussian_filter(bg_img, bg_smooth_sigma)

        movie_imgs = avg_imgs - bg_img[None, :, :]
        movie_imgs[movie_imgs < 0] = 0
    else:
        movie_imgs = avg_imgs.copy()

    if frame_smooth_sigma is not None and frame_smooth_sigma > 0:
        for rep_ind in range(num_frames):
            movie_imgs[rep_ind] = gaussian_filter(
                movie_imgs[rep_ind],
                frame_smooth_sigma,
            )

    if clim is None:
        vmin = np.nanpercentile(movie_imgs, contrast_percentiles[0])
        vmax = np.nanpercentile(movie_imgs, contrast_percentiles[1])

        if vmax <= vmin:
            vmin = np.nanpercentile(movie_imgs, 1)
            vmax = np.nanpercentile(movie_imgs, 99.8)
    else:
        vmin, vmax = clim

    fig, ax = plt.subplots()
    ax.set_axis_off()

    im = ax.imshow(movie_imgs[0], vmin=vmin, vmax=vmax)
    title = ax.set_title(f"rep 0, avg over {num_runs} runs")

    def update(rep_ind):
        im.set_data(movie_imgs[rep_ind])
        title.set_text(f"rep {rep_ind}, avg over {num_runs} runs")
        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval_ms,
        blit=False,
    )

    stem = file_path.stem if file_path.suffix else file_path.name
    gif_path = file_path.with_name(f"{stem}-clean-bgsub-blink.gif")

    fps = max(1, int(1000 / interval_ms))
    anim.save(str(gif_path), writer=PillowWriter(fps=fps))

    plt.close(fig)

    print("Saved clean background-subtracted GIF:", gif_path)
    return gif_path


def _try_get_nv_img_xy(nv):
    """
    Try to extract camera/image pixel coordinate from an NV object.

    Expected output convention:
        x = column pixel
        y = row pixel
    """
    # Direct attributes
    for attr in ["pixel_coords", "img_coords", "image_coords", "camera_coords"]:
        val = getattr(nv, attr, None)
        if val is not None:
            arr = np.asarray(val, dtype=float).ravel()
            if arr.size >= 2:
                return float(arr[0]), float(arr[1])

    # Dictionary-style coords
    coords = getattr(nv, "coords", None)
    if isinstance(coords, dict):
        for key in [
            "pixel",
            "pixels",
            "pixel_coords",
            "img",
            "image",
            "camera",
            "camera_coords",
        ]:
            if key in coords:
                arr = np.asarray(coords[key], dtype=float).ravel()
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1])

    return None


def _coerce_img_coords(nv_list, img_coords=None):
    """
    Return array with shape [num_nvs, 2] in x,y pixel convention.

    If automatic extraction fails, pass:
        img_coords = [[x0, y0], [x1, y1], ...]
    """
    if img_coords is not None:
        arr = np.asarray(img_coords, dtype=float)
        if arr.shape != (len(nv_list), 2):
            raise ValueError(
                f"img_coords must have shape ({len(nv_list)}, 2), got {arr.shape}"
            )
        return arr

    coords = []
    bad = []
    for ind, nv in enumerate(nv_list):
        xy = _try_get_nv_img_xy(nv)
        if xy is None:
            bad.append(ind)
            coords.append([np.nan, np.nan])
        else:
            coords.append(list(xy))

    if len(bad) > 0:
        raise ValueError(
            "Could not automatically find image pixel coordinates for some NVs. "
            f"Example bad indices: {bad[:10]}. "
            "Pass img_coords explicitly as [[x_pixel, y_pixel], ...]."
        )

    return np.asarray(coords, dtype=float)


def _extract_patch(img, x, y, radius):
    """
    Extract patch centered near x,y.
    """
    h, w = img.shape
    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)

    patch = img[y0:y1, x0:x1]
    cy = y - y0
    cx = x - x0

    return patch, (y0, y1, x0, x1), (cy, cx)


def _gaussian_patch_mask(shape, center_yx, radius):
    """
    Smooth mask so pasted NV spots do not have hard square edges.
    """
    h, w = shape
    cy, cx = center_yx

    yy, xx = np.ogrid[:h, :w]
    sigma = max(1.0, radius / 2.0)

    mask = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    max_val = np.nanmax(mask)
    if max_val > 0:
        mask = mask / max_val

    return mask


def _paste_bright_patch(frame, bright_patch, x, y, radius, alpha=1.0, bright_gain=1.0):
    """
    Paste one bright NV patch into a frame.

    alpha:
        0 -> not pasted
        1 -> fully pasted

    bright_gain:
        1.0 normal
        1.2 or 1.5 makes initialized NVs visually brighter
    """
    if alpha <= 0:
        return

    frame_patch, bounds, center_yx = _extract_patch(frame, x, y, radius)
    y0, y1, x0, x1 = bounds

    if frame_patch.size == 0:
        return

    # Safety for edge NVs
    h, w = frame_patch.shape
    bright_patch = bright_patch[:h, :w]

    mask = _gaussian_patch_mask(frame_patch.shape, center_yx, radius)
    alpha_mask = np.clip(alpha * mask, 0, 1)

    target_patch = frame_patch + bright_gain * (bright_patch - frame_patch)

    frame[y0:y1, x0:x1] = (
        frame_patch * (1 - alpha_mask) + target_patch * alpha_mask
    )


def save_cumulative_initialized_movie(
    raw_data,
    file_path,
    mode=None,
    img_coords=None,
    max_reps=None,
    patch_radius=6,
    clim=None,
    interval_ms=120,
    probability_weight=True,
    bright_gain=1.2,
    background_percentile=5,
):
    """
    Save cumulative initialized-NV movie.

    This is better than a normal blink GIF for dmd_block_confirmed data because
    DMD-blocked NVs become dark in the raw images. This function keeps them bright
    once they are confirmed initialized.

    Requires:
        raw_data["img_arrays"]

    img_arrays shape expected:
        [exp, run, step, rep, y, x]

    Movie logic:
        1. Average raw images over runs.
        2. Reconstruct confirmed NV history from counts.
        3. For each NV, build a bright patch from frames where that NV was NV-.
        4. At movie frame rep r, paste that NV patch if it has been confirmed
           by rep r.

    probability_weight=True:
        brightness reflects fraction of runs where the NV is confirmed.

    probability_weight=False:
        once confirmed in any run, the NV is shown fully bright.
    """

    if "img_arrays" not in raw_data:
        print("No img_arrays in raw_data. Skipping cumulative movie.")
        return None

    if mode is None:
        mode = raw_data.get("mode", "old")

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)

    img_arrays = np.asarray(raw_data["img_arrays"], dtype=float)

    # [run, rep, y, x]
    imgs = img_arrays[0, :, 0, :, :, :]

    num_runs, num_reps, img_h, img_w = imgs.shape

    if max_reps is None:
        max_reps = num_reps

    num_frames = min(num_reps, max_reps)

    coords_xy = _coerce_img_coords(nv_list, img_coords=img_coords)

    feedback = reconstruct_confirmed_history(raw_data, mode=mode)

    raw_states = np.asarray(feedback["raw_states"], dtype=bool)
    confirmed_history = np.asarray(feedback["confirmed_history"], dtype=bool)

    # Force cumulative behavior for movie visualization.
    # Shape: [nv, run, rep]
    confirmed_cumulative = np.maximum.accumulate(confirmed_history, axis=2)

    # Fraction confirmed across runs for every NV and rep.
    # Shape: [nv, rep]
    confirmed_prob = np.mean(confirmed_cumulative[:, :, :num_frames], axis=1)

    # Average image over runs for each rep.
    # Shape: [rep, y, x]
    avg_imgs = np.nanmean(imgs[:, :num_frames], axis=0)

    # Dim/background image. This prevents already-blocked NVs from disappearing.
    background_img = np.nanpercentile(
        avg_imgs,
        background_percentile,
        axis=0,
    )

    # Flatten run/rep for building bright patches.
    flat_imgs = imgs.reshape(num_runs * num_reps, img_h, img_w)
    flat_raw_states = raw_states.reshape(num_nvs, num_runs * num_reps)

    # Fallback image if an NV never crosses threshold.
    mean_img = np.nanmean(flat_imgs, axis=0)

    bright_patches = []

    for nv_ind in range(num_nvs):
        x, y = coords_xy[nv_ind]

        if not np.isfinite(x) or not np.isfinite(y):
            patch, _, _ = _extract_patch(mean_img, img_w // 2, img_h // 2, patch_radius)
            bright_patches.append(patch)
            continue

        use = flat_raw_states[nv_ind]

        patches = []
        if np.any(use):
            for img in flat_imgs[use]:
                patch, _, _ = _extract_patch(img, x, y, patch_radius)
                if patch.size > 0:
                    patches.append(patch)

        if len(patches) == 0:
            patch, _, _ = _extract_patch(mean_img, x, y, patch_radius)
        else:
            patch = np.nanmean(np.asarray(patches), axis=0)

        bright_patches.append(patch)

    def make_frame(rep_ind):
        frame = background_img.copy()

        for nv_ind in range(num_nvs):
            p = confirmed_prob[nv_ind, rep_ind]

            if probability_weight:
                alpha = float(p)
            else:
                alpha = 1.0 if p > 0 else 0.0

            if alpha <= 0:
                continue

            x, y = coords_xy[nv_ind]

            _paste_bright_patch(
                frame,
                bright_patches[nv_ind],
                x,
                y,
                patch_radius,
                alpha=alpha,
                bright_gain=bright_gain,
            )

        return frame

    # Choose display limits.
    if clim is None:
        preview0 = make_frame(0)
        preview1 = make_frame(num_frames - 1)

        preview = np.asarray([preview0, preview1])

        vmin = np.nanpercentile(preview, 1)
        vmax = np.nanpercentile(preview, 99.7)
    else:
        vmin, vmax = clim

    fig, ax = plt.subplots()
    ax.set_axis_off()

    first_frame = make_frame(0)
    im = ax.imshow(first_frame, vmin=vmin, vmax=vmax)

    title = ax.set_title(
        f"Attempt 0 | NV$^-$ fraction = {np.mean(confirmed_prob[:, 0]):.3f}"
    )

    def update(frame_ind):
        frame = make_frame(frame_ind)

        im.set_data(frame)
        title.set_text(
            f"Attempt {frame_ind} | NV$^-$ fraction = "
            f"{np.mean(confirmed_prob[:, frame_ind]):.3f}"
        )

        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval_ms,
        blit=False,
    )

    gif_path = f"{file_path}-cumulative-initialized.gif"

    fps = max(1, int(1000 / interval_ms))
    anim.save(gif_path, writer=PillowWriter(fps=fps))

    print("Saved cumulative initialized movie:", gif_path)

    return gif_path


def plot_feedback_profile_summary(raw_data):
    records = raw_data.get("feedback_profile_records", None)

    if records is None or len(records) == 0:
        print("No feedback_profile_records found.")
        return None

    normal_records = [
        rec for rec in records
        if rec.get("phase") in ["normal_feedback", "rep0_start"]
    ]

    if len(normal_records) == 0:
        print("No normal feedback records found.")
        return None

    reps = np.asarray([rec["rep_ind"] for rec in normal_records], dtype=int)

    classify_s = np.asarray(
        [rec.get("classify_s", np.nan) for rec in normal_records],
        dtype=float,
    )
    dmd_s = np.asarray(
        [rec.get("dmd_s", np.nan) for rec in normal_records],
        dtype=float,
    )
    opx_s = np.asarray(
        [rec.get("opx_s", np.nan) for rec in normal_records],
        dtype=float,
    )
    total_s = np.asarray(
        [rec.get("total_callback_s", np.nan) for rec in normal_records],
        dtype=float,
    )
    confirmed = np.asarray(
        [
            np.nan if rec.get("confirmed") is None else rec.get("confirmed")
            for rec in normal_records
        ],
        dtype=float,
    )

    unique_reps = np.unique(reps)

    mean_classify = []
    mean_dmd = []
    mean_opx = []
    mean_total = []
    mean_confirmed = []

    for rep in unique_reps:
        use = reps == rep
        mean_classify.append(np.nanmean(classify_s[use]))
        mean_dmd.append(np.nanmean(dmd_s[use]))
        mean_opx.append(np.nanmean(opx_s[use]))
        mean_total.append(np.nanmean(total_s[use]))
        mean_confirmed.append(np.nanmean(confirmed[use]))

    mean_classify = np.asarray(mean_classify)
    mean_dmd = np.asarray(mean_dmd)
    mean_opx = np.asarray(mean_opx)
    mean_total = np.asarray(mean_total)
    mean_confirmed = np.asarray(mean_confirmed)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(unique_reps, mean_classify, "o-", label="classify")
    axes[0].plot(unique_reps, mean_dmd, "o-", label="DMD update")
    axes[0].plot(unique_reps, mean_opx, "o-", label="OPX stream")
    axes[0].plot(unique_reps, mean_total, "o-", label="total callback")

    axes[0].set_ylabel("Time per callback (s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(unique_reps, mean_confirmed, "o-")
    axes[1].set_xlabel("Rep index")
    axes[1].set_ylabel("Mean confirmed NVs")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Feedback-loop profiling", fontsize=14)
    plt.tight_layout()

    return fig
# =============================================================================
# Main routine
# =============================================================================
def main(
    nv_list,
    num_reps,
    num_runs,
    mode="old",
    dmd_indices=None,
    dmd_radius_px=8,
    dmd_plane=230,
    confirm_margin_counts=0.0,
    dmd_settle_s=0.001,
    save_images=True,
    save_images_avg_reps=False,
    save_data=True,
    save_fig=True,
    save_image_frames=False,
    save_movie=True,
    reset_dmd_on_exit=True,
    verbose=True,
    take_final_check_image=True,
    track_rep_timing=True,
    profile_feedback=True,
):
    """
    Run conditional charge initialization in one of three modes.

    mode:
        "old":
            exactly old conditional init, no DMD

        "dmd_all_on":
            same old conditional init, but DMD kept pass-all/block-none

        "dmd_block_confirmed":
            same sequence/reps/runs, DMD blocks confirmed NV- sites
    """

    valid_modes = ["old", "dmd_all_on", "dmd_block_confirmed"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}")

    seq_file = "charge_state_conditional_init.py"
    num_steps = 1
    
    num_reps_analysis = int(num_reps)

    if take_final_check_image:
        final_check_rep_ind = num_reps_analysis
        num_reps_total = num_reps_analysis + 1
    else:
        final_check_rep_ind = None
        num_reps_total = num_reps_analysis

    nv_run_list = _copy_nv_list_with_margin(
        nv_list,
        confirm_margin_counts=confirm_margin_counts,
    )

    num_nvs = len(nv_run_list)
    thresholds = _get_thresholds(nv_run_list)
    dmd_indices = _prepare_dmd_indices(num_nvs, dmd_indices)

    print("\n=== conditional charge initialization comparison ===")
    print("mode:", mode)
    print("num NVs:", num_nvs)
    print("num_reps:", num_reps)
    print("num_runs:", num_runs)
    print("save_images:", save_images)
    print("confirm_margin_counts:", confirm_margin_counts)
    print("threshold range:", np.nanmin(thresholds), np.nanmax(thresholds))
    print("num_reps analysis:", num_reps_analysis)
    print("num_reps total:", num_reps_total)
    print("take_final_check_image:", take_final_check_image)
    print("final_check_rep_ind:", final_check_rep_ind)
    
    feedback_profile_records = [] if profile_feedback else None
    rep_timing_records = []

    if mode == "old":
        charge_prep_fn = base_routine.charge_prep_no_verification_skip_first_rep

    elif mode == "dmd_all_on":
        charge_prep_fn = make_dmd_all_on_charge_prep_fn(
            num_nvs=num_nvs,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            use_dmd=True,
            dmd_settle_s=dmd_settle_s,
            verbose=verbose,
        )

    elif mode == "dmd_block_confirmed":
        charge_prep_fn = make_dmd_block_confirmed_charge_prep_fn(
            num_nvs=num_nvs,
            dmd_indices=dmd_indices,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            use_dmd=True,
            dmd_settle_s=dmd_settle_s,
            verbose=verbose,
            profile_records=feedback_profile_records,
        )

    if take_final_check_image:
        charge_prep_fn = make_final_check_charge_prep_fn(
            charge_prep_fn,
            num_nvs=num_nvs,
            final_check_rep_ind=final_check_rep_ind,
            mode=mode,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            dmd_settle_s=dmd_settle_s,
            verbose=verbose,
            profile_records=feedback_profile_records,
        )

    if track_rep_timing:
        charge_prep_fn = make_timed_charge_prep_fn(
            charge_prep_fn,
            timing_records=rep_timing_records,
            verbose=False,
        )
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        # Same exact args as old conditional init.
        ion_coords_list = widefield.get_coords_list(
            nv_run_list,
            VirtualLaserKey.ION,
        )

        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(
                nv_run_list,
                VirtualLaserKey.CHARGE_POL,
            )
        )

        seq_args = [
            ion_coords_list,
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
        ]

        seq_args_string = tb.encode_seq_args(seq_args)

        # Same exact pattern as old conditional init:
        # reps are sent as stream_load(..., num_reps)
        pulse_gen.stream_load(
            seq_file,
            seq_args_string,
            num_reps_total,
        )

    try:
        t_experiment0 = time.perf_counter()
        raw_data = base_routine.main(
            nv_run_list,
            num_steps,
            num_reps_total,
            num_runs,
            run_fn=run_fn,
            save_images=save_images,
            save_images_avg_reps=save_images_avg_reps,
            charge_prep_fn=charge_prep_fn,
            num_exps=1,
            uwave_ind_list=[],
        )

        t_experiment1 = time.perf_counter()
        experiment_wall_s = float(t_experiment1 - t_experiment0)

    finally:
        if reset_dmd_on_exit and mode.startswith("dmd"):
            try:
                _dmd_pass_all_block_none(
                    dmd_radius_px=dmd_radius_px,
                    dmd_plane=dmd_plane,
                    use_dmd=True,
                    dmd_settle_s=dmd_settle_s,
                )
            except Exception:
                print("Could not reset DMD:")
                print(traceback.format_exc())

    raw_data |= {
        "mode": mode,
        "thresholds": thresholds,
        "dmd_indices": dmd_indices,
        "dmd_radius_px": dmd_radius_px,
        "dmd_plane": dmd_plane,
        "confirm_margin_counts": confirm_margin_counts,
        "timestamp": dm.get_time_stamp(),
        "img_array-units": "photons",
        "num_reps_analysis": int(num_reps_analysis),
        "num_reps_total": int(num_reps_total),
        "take_final_check_image": bool(take_final_check_image),
        "final_check_rep_ind": None
        if final_check_rep_ind is None
        else int(final_check_rep_ind),
        "rep_timing_records": rep_timing_records,
        "feedback_profile_records": feedback_profile_records,
        "experiment_wall_s": experiment_wall_s,
    }


    # Save full raw_data including nv_list and images.
    timestamp = raw_data["timestamp"]
    repr_nv_sig = widefield.get_repr_nv_sig(nv_run_list)
    repr_nv_name = repr_nv_sig.name

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        f"{repr_nv_name}-{mode}",
    )

    keys_to_compress = ["counts", "thresholds", "dmd_indices"]
    if save_images and "img_arrays" in raw_data:
        keys_to_compress.append("img_arrays")

    dm.save_raw_data(
        raw_data,
        file_path,
        keys_to_compress=keys_to_compress,
    )

    print("Saved raw data:", file_path)
    
    try:
        fig = process_and_plot(raw_data, mode=mode)
    except Exception:
        print(traceback.format_exc())
        fig = None

    if save_fig:
        try:
            fig_timing = plot_rep_timing_summary(raw_data)
            if fig_timing is not None:
                dm.save_figure(fig_timing, _append_to_file_path(file_path, "rep-timing"))
                print("Saved rep timing plot:", f"{file_path}-rep-timing")
        except Exception:
            print("Could not save rep timing plot:")
            print(traceback.format_exc())
            
        try:
            fig_profile = plot_feedback_profile_summary(raw_data)
            if fig_profile is not None:
                dm.save_figure(fig_profile, _append_to_file_path(file_path, "rep-timing"))
                print("Saved feedback profile plot:", f"{file_path}-feedback-profile")
        except Exception:
            print("Could not save feedback profile plot:")
            print(traceback.format_exc())


        try:
            fig_final = plot_final_check_image(raw_data)
            if fig_final is not None:
                dm.save_figure(fig_final, _append_to_file_path(file_path, "final-check-image"))
                print("Saved final-check image:", f"{file_path}-final-check-image")
        except Exception:
            print("Could not save final-check image:")
            print(traceback.format_exc())

    # if save_images and save_image_frames:
    #     try:
    #         save_avg_rep_images(
    #             raw_data,
    #             file_path,
    #             reps_to_save=list(range(min(num_reps, 10))),
    #             clim=None,
    #         )
    #     except Exception:
    #         print("Could not save avg rep images:")
    #         print(traceback.format_exc())

    if save_images and save_movie:
        try:
            save_blink_gif(
                raw_data,
                file_path,
                max_reps=min(num_reps, 20),
                clim=None,
                interval_ms=250,
            )
        except Exception:
            print("Could not save blink GIF:")
            print(traceback.format_exc())

    tb.reset_cfm()

    return raw_data


# =============================================================================
# Convenience comparison runner
# =============================================================================
def run_old_and_dmd_compare(
    nv_list,
    num_reps=20,
    num_runs=10,
    dmd_indices=None,
    dmd_radius_px=8,
    dmd_plane=230,
    confirm_margin_counts=0.0,
    save_images=True,
):
    """
    Run old and DMD-block-confirmed experiments separately with same settings.
    """

    raw_old = main(
        nv_list,
        num_reps=num_reps,
        num_runs=num_runs,
        mode="old",
        dmd_indices=dmd_indices,
        dmd_radius_px=dmd_radius_px,
        dmd_plane=dmd_plane,
        confirm_margin_counts=confirm_margin_counts,
        save_images=save_images,
        save_data=True,
        save_fig=True,
        save_image_frames=True,
        save_movie=True,
        verbose=True,
    )

    raw_dmd = main(
        nv_list,
        num_reps=num_reps,
        num_runs=num_runs,
        mode="dmd_block_confirmed",
        dmd_indices=dmd_indices,
        dmd_radius_px=dmd_radius_px,
        dmd_plane=dmd_plane,
        confirm_margin_counts=confirm_margin_counts,
        save_images=save_images,
        save_data=True,
        save_fig=True,
        save_image_frames=True,
        save_movie=True,
        verbose=True,
    )

    return raw_old, raw_dmd



def get_final_check_charge_change_indices(
    raw_data,
    mode=None,
    print_details=True,
):
    """
    Compare the cumulative adaptive-confirmation state with the independent
    final-check image.

    Regular per-NV thresholds are used for the final image:

        count > threshold  -> NV-
        count <= threshold -> NV0

    Returns
    -------
    result : dict

        lost_nv_inds_by_run:
            Previously confirmed NV-, final-check NV0.

        retained_nv_inds_by_run:
            Previously confirmed NV-, final-check NV-.

        gained_nv_inds_by_run:
            Not previously confirmed, final-check NV-.

        final_nv0_inds_by_run:
            Every NV classified NV0 in the final image.

        lost_details_by_run:
            Counts and thresholds for each lost NV.
    """

    if mode is None:
        mode = raw_data.get("mode", "dmd_block_confirmed")

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)

    # --------------------------------------------------------------
    # Regular calibrated thresholds.
    # Do not add confirm_margin_counts here.
    # --------------------------------------------------------------
    thresholds = np.asarray(
        [float(nv.threshold) for nv in nv_list],
        dtype=float,
    )

    final_rep_ind = raw_data.get("final_check_rep_ind", None)
    if final_rep_ind is None:
        raise ValueError(
            "raw_data has no final_check_rep_ind. "
            "The measurement must include take_final_check_image=True."
        )

    final_rep_ind = int(final_rep_ind)

    counts_all = np.asarray(raw_data["counts"], dtype=float)

    if counts_all.ndim != 5:
        raise ValueError(
            "Expected counts shape [exp, nv, run, step, rep], "
            f"got {counts_all.shape}."
        )

    # [nv, run]
    final_counts = counts_all[
        0,
        :,
        :,
        0,
        final_rep_ind,
    ]

    num_runs = final_counts.shape[1]

    # --------------------------------------------------------------
    # Final image classification using regular threshold.
    # --------------------------------------------------------------
    final_nvm = final_counts > thresholds[:, None]
    final_nv0 = ~final_nvm

    # --------------------------------------------------------------
    # Reconstruct persistent/ever-confirmed adaptive state.
    #
    # This uses the same logic as your existing adaptive analysis:
    # once an NV is confirmed, it remains confirmed.
    # --------------------------------------------------------------
    feedback = reconstruct_confirmed_history(
        raw_data,
        mode=mode,
    )

    confirmed_history = np.asarray(
        feedback["confirmed_history"],
        dtype=bool,
    )

    # Last adaptive repetition only, before the extra final check.
    previous_confirmed = confirmed_history[:, :, -1]

    if previous_confirmed.shape != final_nvm.shape:
        raise ValueError(
            "Previous/final state shape mismatch: "
            f"{previous_confirmed.shape} versus {final_nvm.shape}."
        )

    # --------------------------------------------------------------
    # Transition masks
    # --------------------------------------------------------------
    lost_mask = previous_confirmed & final_nv0
    retained_mask = previous_confirmed & final_nvm
    gained_mask = (~previous_confirmed) & final_nvm
    stayed_unconfirmed_mask = (~previous_confirmed) & final_nv0

    lost_nv_inds_by_run = []
    retained_nv_inds_by_run = []
    gained_nv_inds_by_run = []
    final_nv0_inds_by_run = []
    lost_details_by_run = []

    for run_ind in range(num_runs):
        lost_inds = np.where(lost_mask[:, run_ind])[0].astype(int)
        retained_inds = np.where(retained_mask[:, run_ind])[0].astype(int)
        gained_inds = np.where(gained_mask[:, run_ind])[0].astype(int)
        final_nv0_inds = np.where(final_nv0[:, run_ind])[0].astype(int)

        lost_nv_inds_by_run.append(lost_inds.tolist())
        retained_nv_inds_by_run.append(retained_inds.tolist())
        gained_nv_inds_by_run.append(gained_inds.tolist())
        final_nv0_inds_by_run.append(final_nv0_inds.tolist())

        run_details = []

        for nv_ind in lost_inds:
            run_details.append(
                {
                    "nv_ind": int(nv_ind),
                    "threshold": float(thresholds[nv_ind]),
                    "final_count": float(
                        final_counts[nv_ind, run_ind]
                    ),
                    "count_below_threshold": float(
                        thresholds[nv_ind]
                        - final_counts[nv_ind, run_ind]
                    ),
                }
            )

        lost_details_by_run.append(run_details)

    previous_confirmed_by_run = np.sum(
        previous_confirmed,
        axis=0,
    )
    final_nvm_by_run = np.sum(final_nvm, axis=0)
    lost_by_run = np.sum(lost_mask, axis=0)
    retained_by_run = np.sum(retained_mask, axis=0)
    gained_by_run = np.sum(gained_mask, axis=0)

    result = {
        "mode": mode,
        "num_nvs": int(num_nvs),
        "num_runs": int(num_runs),
        "final_check_rep_ind": int(final_rep_ind),

        "thresholds": thresholds,
        "final_counts": final_counts,

        "previous_confirmed_mask": previous_confirmed,
        "final_nvm_mask": final_nvm,
        "final_nv0_mask": final_nv0,

        "lost_mask": lost_mask,
        "retained_mask": retained_mask,
        "gained_mask": gained_mask,
        "stayed_unconfirmed_mask": stayed_unconfirmed_mask,

        "lost_nv_inds_by_run": lost_nv_inds_by_run,
        "retained_nv_inds_by_run": retained_nv_inds_by_run,
        "gained_nv_inds_by_run": gained_nv_inds_by_run,
        "final_nv0_inds_by_run": final_nv0_inds_by_run,
        "lost_details_by_run": lost_details_by_run,

        "previous_confirmed_by_run": previous_confirmed_by_run,
        "final_nvm_by_run": final_nvm_by_run,
        "lost_by_run": lost_by_run,
        "retained_by_run": retained_by_run,
        "gained_by_run": gained_by_run,
    }

    if print_details:
        print("\n=== Final-check charge-state changes ===")
        print("Regular per-NV thresholds used.")
        print("Final-check rep:", final_rep_ind)

        for run_ind in range(num_runs):
            print("\n" + "-" * 60)
            print(f"Run {run_ind}")
            print(
                "Previously confirmed NV-:",
                int(previous_confirmed_by_run[run_ind]),
            )
            print(
                "Final-check NV-:",
                int(final_nvm_by_run[run_ind]),
            )
            print(
                "Retained confirmed NV-:",
                int(retained_by_run[run_ind]),
            )
            print(
                "Lost NV- -> NV0:",
                int(lost_by_run[run_ind]),
            )
            print(
                "New/unconfirmed -> final NV-:",
                int(gained_by_run[run_ind]),
            )
            print(
                "Lost NV indices:",
                lost_nv_inds_by_run[run_ind],
            )

    return result

def plot_lost_charge_heatmap(change_result):
    """
    Rows = run index
    Columns = NV index

    White/low = retained or not previously confirmed
    Red/high = previously confirmed NV- but final-check NV0
    """

    lost_mask = np.asarray(
        change_result["lost_mask"],
        dtype=bool,
    )  # [nv, run]

    # Convert to [run, nv] for plotting.
    lost_map = lost_mask.T.astype(float)

    num_runs, num_nvs = lost_map.shape

    fig, ax = plt.subplots(figsize=(14, 4.5))

    im = ax.imshow(
        lost_map,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("NV index")
    ax.set_ylabel("Run index")
    ax.set_title("NVs that lost charge at final check")

    ax.set_yticks(np.arange(num_runs))
    ax.set_yticklabels(np.arange(num_runs))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Not lost", "NV$^- \\rightarrow$ NV$^0$"])

    return fig


def plot_loss_count_by_nv(change_result):
    """
    Plot how many runs each NV lost its charge state.
    """

    lost_mask = np.asarray(
        change_result["lost_mask"],
        dtype=bool,
    )  # [nv, run]

    loss_count_by_nv = np.sum(lost_mask, axis=1)
    num_runs = lost_mask.shape[1]

    nv_inds = np.arange(loss_count_by_nv.size)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(
        nv_inds,
        loss_count_by_nv,
        color=kpl.KplColors.RED,
        width=1.0,
    )

    ax.set_xlabel("NV index")
    ax.set_ylabel("Number of runs lost")
    ax.set_title(
        f"Final-check charge loss frequency across {num_runs} runs"
    )

    ax.set_ylim(0, num_runs + 0.25)
    ax.set_yticks(np.arange(num_runs + 1))
    ax.grid(True, axis="y", alpha=0.3)

    return fig


def print_ranked_lost_nvs(change_result):
    lost_mask = np.asarray(
        change_result["lost_mask"],
        dtype=bool,
    )

    loss_count_by_nv = np.sum(lost_mask, axis=1)
    num_runs = lost_mask.shape[1]

    ranked_inds = np.argsort(loss_count_by_nv)[::-1]

    print("\n=== Ranked unstable NVs ===")

    for nv_ind in ranked_inds:
        count = int(loss_count_by_nv[nv_ind])

        if count == 0:
            break

        lost_runs = np.where(
            lost_mask[nv_ind]
        )[0].astype(int).tolist()

        print(
            f"NV {nv_ind}: lost {count}/{num_runs} runs, "
            f"runs={lost_runs}"
        )

    # --------------------------------------------------------------
    # Selected NV index lists
    # --------------------------------------------------------------

    # Lost in at least 2 runs.
    lost_at_least_2_inds = np.where(
        loss_count_by_nv >= 2
    )[0].astype(int).tolist()

    # Lost in at least 3 runs.
    lost_at_least_3_inds = np.where(
        loss_count_by_nv >= 3
    )[0].astype(int).tolist()

    # Lost in exactly 2 runs.
    lost_exactly_2_inds = np.where(
        loss_count_by_nv == 2
    )[0].astype(int).tolist()

    # Lost in exactly 3 runs.
    lost_exactly_3_inds = np.where(
        loss_count_by_nv == 3
    )[0].astype(int).tolist()

    # Lost in every run.
    lost_every_run_inds = np.where(
        loss_count_by_nv == num_runs
    )[0].astype(int).tolist()

    print("\n=== NV index lists ===")

    print(
        f"Lost in at least 2 runs "
        f"({len(lost_at_least_2_inds)} NVs):"
    )
    print(lost_at_least_2_inds)

    print(
        f"\nLost in at least 3 runs "
        f"({len(lost_at_least_3_inds)} NVs):"
    )
    print(lost_at_least_3_inds)

    print(
        f"\nLost in exactly 2 runs "
        f"({len(lost_exactly_2_inds)} NVs):"
    )
    print(lost_exactly_2_inds)

    print(
        f"\nLost in exactly 3 runs "
        f"({len(lost_exactly_3_inds)} NVs):"
    )
    print(lost_exactly_3_inds)

    print(
        f"\nLost in every run "
        f"({len(lost_every_run_inds)} NVs):"
    )
    print(lost_every_run_inds)

    return {
        "loss_count_by_nv": loss_count_by_nv,
        "lost_at_least_2_inds": lost_at_least_2_inds,
        "lost_at_least_3_inds": lost_at_least_3_inds,
        "lost_exactly_2_inds": lost_exactly_2_inds,
        "lost_exactly_3_inds": lost_exactly_3_inds,
        "lost_every_run_inds": lost_every_run_inds,
    }


def plot_repeatedly_lost_heatmap(
    change_result,
    min_loss_count=2,
):
    lost_mask = np.asarray(
        change_result["lost_mask"],
        dtype=bool,
    )  # [nv, run]

    loss_count = np.sum(lost_mask, axis=1)

    selected_inds = np.where(
        loss_count >= int(min_loss_count)
    )[0]

    if selected_inds.size == 0:
        print("No NVs satisfy the requested minimum loss count.")
        return None

    selected_map = lost_mask[selected_inds].T.astype(float)

    num_runs = selected_map.shape[0]

    fig_width = max(8, 0.25 * selected_inds.size)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))

    im = ax.imshow(
        selected_map,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("NV index")
    ax.set_ylabel("Run index")
    ax.set_title(
        f"NVs lost in at least {min_loss_count} of {num_runs} runs"
    )

    ax.set_xticks(np.arange(selected_inds.size))
    ax.set_xticklabels(
        selected_inds,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticks(np.arange(num_runs))
    ax.set_yticklabels(np.arange(num_runs))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Retained", "Lost"])

    return fig


def plot_loss_probability_spatial(
    raw_data,
    change_result,
    img_coords=None,
    marker_size=45,
):
    lost_mask = np.asarray(
        change_result["lost_mask"],
        dtype=bool,
    )

    loss_probability = np.mean(lost_mask, axis=1)

    coords_xy = _coerce_img_coords(
        raw_data["nv_list"],
        img_coords=img_coords,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    sc = ax.scatter(
        coords_xy[:, 0],
        coords_xy[:, 1],
        c=loss_probability,
        s=marker_size,
        cmap="Reds",
        vmin=0,
        vmax=1,
        edgecolors=kpl.KplColors.GRAY,
        linewidths=0.4,
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(
        "Fraction of runs with NV$^- \\rightarrow$ NV$^0$"
    )

    ax.set_xlabel("Camera x pixel")
    ax.set_ylabel("Camera y pixel")
    ax.set_title("Spatial distribution of final-check charge loss")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    return fig

def print_selected_nv_final_counts(
    raw_data,
    nv_inds=(8, 303, 364, 422, 463, 536),
):
    """
    Print the regular threshold and final-check count across all runs
    for selected NVs.

    Counts shape:
        counts[exp, nv, run, step, rep]

    Returns
    -------
    result : dict
        Contains thresholds, final counts, states, and count-threshold
        differences for the selected NVs.
    """

    nv_inds = np.asarray(nv_inds, dtype=int)

    counts = np.asarray(
        raw_data["counts"],
        dtype=float,
    )

    if counts.ndim != 5:
        raise ValueError(
            "Expected counts[exp, nv, run, step, rep], "
            f"got shape {counts.shape}."
        )

    final_rep_ind = raw_data.get(
        "final_check_rep_ind",
        None,
    )

    if final_rep_ind is None:
        raise ValueError(
            "No final_check_rep_ind found in raw_data."
        )

    final_rep_ind = int(final_rep_ind)

    # --------------------------------------------------------------
    # The saved thresholds include confirm_margin_counts.
    # Recover the original regular per-NV thresholds.
    # --------------------------------------------------------------
    feedback_thresholds = np.asarray(
        raw_data["thresholds"],
        dtype=float,
    )

    confirm_margin = float(
        raw_data.get("confirm_margin_counts", 0.0)
    )

    regular_thresholds = (
        feedback_thresholds - confirm_margin
    )

    num_nvs = counts.shape[1]
    num_runs = counts.shape[2]

    if np.any(nv_inds < 0) or np.any(nv_inds >= num_nvs):
        raise IndexError(
            f"NV indices must lie between 0 and {num_nvs - 1}."
        )

    # [selected_nv, run]
    final_counts = counts[
        0,
        nv_inds,
        :,
        0,
        final_rep_ind,
    ]

    selected_regular_thresholds = regular_thresholds[
        nv_inds
    ]

    selected_feedback_thresholds = feedback_thresholds[
        nv_inds
    ]

    # [selected_nv, run]
    final_nvm = (
        final_counts
        > selected_regular_thresholds[:, None]
    )

    delta_from_threshold = (
        final_counts
        - selected_regular_thresholds[:, None]
    )

    print("\n=== Selected NV final-check counts ===")
    print("Final-check rep:", final_rep_ind)
    print("Number of runs:", num_runs)
    print("Confirmation margin:", confirm_margin)
    print(
        "Classification: count > regular threshold -> NV-"
    )

    for local_ind, nv_ind in enumerate(nv_inds):
        regular_threshold = float(
            selected_regular_thresholds[local_ind]
        )

        feedback_threshold = float(
            selected_feedback_thresholds[local_ind]
        )

        print("\n" + "=" * 72)
        print(f"NV {nv_ind}")
        print(
            f"Regular threshold:  {regular_threshold:.3f}"
        )
        print(
            f"Feedback threshold: {feedback_threshold:.3f}"
        )

        for run_ind in range(num_runs):
            count = float(
                final_counts[local_ind, run_ind]
            )

            delta = float(
                delta_from_threshold[
                    local_ind,
                    run_ind,
                ]
            )

            state = (
                "NV-"
                if final_nvm[local_ind, run_ind]
                else "NV0"
            )

            print(
                f"Run {run_ind}: "
                f"final count = {count:8.3f}, "
                f"count - threshold = {delta:+8.3f}, "
                f"state = {state}"
            )

        print(
            "Mean final count: "
            f"{np.mean(final_counts[local_ind]):.3f}"
        )

        print(
            "Final NV- runs: "
            f"{int(np.sum(final_nvm[local_ind]))}"
            f"/{num_runs}"
        )

    result = {
        "nv_inds": nv_inds.tolist(),
        "final_check_rep_ind": final_rep_ind,
        "confirm_margin_counts": confirm_margin,
        "regular_thresholds": (
            selected_regular_thresholds.tolist()
        ),
        "feedback_thresholds": (
            selected_feedback_thresholds.tolist()
        ),
        "final_counts": final_counts.tolist(),
        "delta_from_regular_threshold": (
            delta_from_threshold.tolist()
        ),
        "final_nvm_mask": final_nvm.tolist(),
    }

    return result

if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example plotting existing file:
    raw_data = dm.get_raw_data(
        # file_stem="2026_07_19-01_02_13-qnami-nv0_2026_02_20-dmd_block_confirmed", 
        file_stem = "2026_07_21-16_15_53-qnami-nv0_2026_02_20-dmd_block_confirmed",
        load_npz=True)

    timestamp = raw_data["timestamp"]
    file_path = dm.get_file_path(__file__, timestamp, "movie")
    
    change_result =  get_final_check_charge_change_indices(
    raw_data,
    mode="dmd_block_confirmed",
    )
    # plot_lost_charge_heatmap(change_result)
    # plot_loss_count_by_nv(change_result)
    # print_ranked_lost_nvs(change_result)
    # plot_repeatedly_lost_heatmap(change_result, min_loss_count=4)
    # plot_loss_probability_spatial(raw_data, change_result)
    # selected_nv_result = print_selected_nv_final_counts(
    # raw_data, nv_inds=[8, 303, 364, 422, 463, 536])
    
    summary, fig = analyze_and_plot_final_check(
        raw_data,
        mode=raw_data.get(
            "mode",
            "dmd_block_confirmed",
        ),
        # Separately flag losses within five counts of threshold.
        borderline_window_counts=1.0,

        save_data=False,
        save_fig=True,
    )
    
    # fig_timing = plot_rep_timing_summary(raw_data)
    # file_path = dm.get_file_path(__file__, timestamp, "rep-timing")
    # dm.save_figure(fig_timing, _append_to_file_path(file_path, "rep-timing"))
    # feedback_profile = plot_feedback_profile_summary(raw_data)
    # file_path = dm.get_file_path(__file__, timestamp, "feedback-profile")
    # dm.save_figure(feedback_profile, _append_to_file_path(file_path, "feedback-profile"))
    kpl.show(block=True)
    
    # save_blink_gif(
    # raw_data,
    # file_path,
    # max_reps=20,
    # clim=None,
    # interval_ms=120,
    # patch_radius=6,
    # probability_weight=True,
    # bright_gain=1.2,
    # )
    # save_blink_gif(
    #     raw_data,
    #     file_path,
    #     max_reps=20,
    #     clim=None,
    #     interval_ms=500,
    # )
    # save_blink_gif(
    #     raw_data,
    #     file_path,
    #     max_reps=20,
    #     interval_ms=500,
    #     background_percentile=5,
    #     bg_smooth_sigma=2,
    #     frame_smooth_sigma=0,
    #     contrast_percentiles=(60, 99.8),
    #     )
    # save_blink_gif(
    #     raw_data,
    #     file_path,
    #     max_reps=100,
    #     clim=None,
    #     interval_ms=10,
    # )
    
    # save_cumulative_initialized_movie(
    #     raw_data,
    #     file_path,
    #     mode=raw_data.get("mode", "dmd_block_confirmed"),
    #     max_reps=100,
    #     patch_radius=2,
    #     clim=None,
    #     interval_ms=500,
    #     probability_weight=True,
    #     bright_gain=1.0,
    # )

    # fig = process_and_plot(raw_data, mode=raw_data.get("mode", "old"), save_fig=True)
    kpl.show(block=True)

    # pass