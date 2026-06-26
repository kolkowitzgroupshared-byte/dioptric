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
):
    """
    Custom charge_prep_fn compatible with base_routine.main.

    Same idea as old conditional init:
        rep 0: no charge prep, only the sequence's rep0 behavior
        rep > 0: send _cache_target_list to OPX

    Difference:
        keeps persistent confirmed_nvm mask.
        once NV is confirmed NV-, DMD blocks it and OPX stops targeting it.
    """

    dmd_indices = _prepare_dmd_indices(num_nvs, dmd_indices)
    pulse_gen = tb.get_server_pulse_gen()
    dmd = tb.get_server_dmd() if use_dmd else None

    confirmed_nvm = np.zeros(num_nvs, dtype=bool)
    last_confirmed_mask = {"mask": None}
    run_counter = {"run_ind": -1}

    def update_dmd_if_needed(confirmed_mask, force=False):
        confirmed_mask = np.asarray(confirmed_mask, dtype=bool)

        if (
            not force
            and last_confirmed_mask["mask"] is not None
            and np.array_equal(confirmed_mask, last_confirmed_mask["mask"])
        ):
            return

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

        last_confirmed_mask["mask"] = confirmed_mask.copy()

    def charge_prep_fn(rep_ind, nv_list, initial_states_list=None):
        nonlocal confirmed_nvm

        # Start of each run.
        if rep_ind == 0:
            run_counter["run_ind"] += 1
            confirmed_nvm[:] = False
            last_confirmed_mask["mask"] = None

            # DMD pass all / block no NV sites.
            update_dmd_if_needed(confirmed_nvm, force=True)

            if verbose:
                print(
                    f"[DMD block confirmed] run {run_counter['run_ind']}, "
                    f"rep {rep_ind}: start run, no NVs blocked"
                )

            # Match old skip-first-rep behavior: no target list on rep 0.
            return

        # Update confirmation from previous readout.
        if initial_states_list is not None:
            states = np.asarray(initial_states_list, dtype=bool)
            active_prev = ~confirmed_nvm
            newly_confirmed = active_prev & states
            confirmed_nvm[newly_confirmed] = True
        else:
            newly_confirmed = np.zeros(num_nvs, dtype=bool)

        active_mask = ~confirmed_nvm

        # DMD blocks confirmed NVs.
        update_dmd_if_needed(confirmed_nvm, force=False)

        # OPX targets only unconfirmed NVs.
        pulse_gen.insert_input_stream(
            "_cache_target_list",
            active_mask.astype(bool).tolist(),
        )

        if verbose:
            print(
                f"[DMD block confirmed] run {run_counter['run_ind']}, "
                f"rep {rep_ind}: "
                f"new={int(np.sum(newly_confirmed))}, "
                f"confirmed={int(np.sum(confirmed_nvm))}/{num_nvs}, "
                f"active={int(np.sum(active_mask))}/{num_nvs}"
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


def process_and_plot(raw_data, mode=None, mean_val=None, save_fig=False):
    if mode is None:
        mode = raw_data.get("mode", "old")

    nv_list = raw_data["nv_list"]
    num_nvs = len(nv_list)
    num_runs = int(raw_data["num_runs"])
    num_reps = int(raw_data["num_reps"])

    feedback = reconstruct_confirmed_history(raw_data, mode=mode)
    raw_data["feedback_summary"] = feedback

    avg_fraction = feedback["avg_fraction"]
    ste_fraction = feedback["ste_fraction"]

    reps_vals = np.arange(num_reps)

    xlim = min(11, num_reps)
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

        rep_file_path = f"{file_path}-avg-img-rep{rep_ind}"
        dm.save_figure(fig, rep_file_path)

        saved_figs.append(fig)

    return saved_figs


from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter


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


from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt


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
    save_image_frames=True,
    save_movie=True,
    reset_dmd_on_exit=True,
    verbose=True,
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
            num_reps,
        )

    try:
        raw_data = base_routine.main(
            nv_run_list,
            num_steps,
            num_reps,
            num_runs,
            run_fn=run_fn,
            save_images=save_images,
            save_images_avg_reps=save_images_avg_reps,
            charge_prep_fn=charge_prep_fn,
            num_exps=1,
            uwave_ind_list=[],
        )

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
    }

    try:
        fig = process_and_plot(raw_data, mode=mode)
    except Exception:
        print(traceback.format_exc())
        fig = None

    if save_data:
        timestamp = raw_data["timestamp"]
        repr_nv_sig = widefield.get_repr_nv_sig(nv_run_list)
        repr_nv_name = repr_nv_sig.name

        file_path = dm.get_file_path(
            __file__,
            timestamp,
            f"{repr_nv_name}-{mode}",
        )

        # Save full raw_data including nv_list and images.
        # If this becomes too large, set save_images=False or save fewer runs.
        keys_to_compress = ["counts", "thresholds", "dmd_indices"]
        if save_images and "img_arrays" in raw_data:
            keys_to_compress.append("img_arrays")

        dm.save_raw_data(
            raw_data,
            file_path,
            keys_to_compress=keys_to_compress,
        )

        print("Saved raw data:", file_path)

        if save_fig and fig is not None:
            dm.save_figure(fig, file_path)

        if save_images and save_image_frames:
            try:
                save_avg_rep_images(
                    raw_data,
                    file_path,
                    reps_to_save=list(range(min(num_reps, 10))),
                    clim=None,
                )
            except Exception:
                print("Could not save avg rep images:")
                print(traceback.format_exc())

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


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example plotting existing file:
    raw_data = dm.get_raw_data(
        # file_stem="2026_06_23-18_41_07-qnami-nv0_2026_02_20-dmd_block_confirmed", 
        # file_stem="2026_06_23-19_23_49-qnami-nv0_2026_02_20-old", 
        # file_stem="2026_06_23-21_03_10-qnami-nv0_2026_02_20-dmd_block_confirmed", 
        # file_stem="2026_06_23-21_23_53-qnami-nv0_2026_02_20-dmd_block_confirmed", 
        file_stem="2026_06_23-21_58_45-qnami-nv0_2026_02_20-dmd_block_confirmed", 
        
        load_npz=True)
    timestamp = raw_data["timestamp"]

    file_path = dm.get_file_path(__file__,timestamp,"movie")
    
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
    save_blink_gif(
    raw_data,
    file_path,
    max_reps=20,
    interval_ms=500,
    background_percentile=5,
    bg_smooth_sigma=2,
    frame_smooth_sigma=0,
    contrast_percentiles=(60, 99.8),
    )
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
    #     interval_ms=1000,
    #     probability_weight=True,
    #     bright_gain=1.0,
    # )

    # fig = process_and_plot(raw_data, mode=raw_data.get("mode", "old"), save_fig=True)
    kpl.show(block=True)

    # pass