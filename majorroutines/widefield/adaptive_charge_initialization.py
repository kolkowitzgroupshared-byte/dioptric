# -*- coding: utf-8 -*-
"""
Adaptive active-set NV- charge initialization.

Goal:
    Initialize many NVs into NV- while minimizing destructive charge-state readout.

Protocol:
    1. Initial readout of all NVs.
    2. Confirm NVs already in NV-.
    3. Remove confirmed NV- sites from future charge-prep and readout.
    4. Repeat charge-polarization + readout only on remaining active NVs.

Created: 2026-06-12
"""

import json
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


# -----------------------------
# Helper functions
# -----------------------------

def subset_nv_list(nv_list, inds):
    """Return nv_list subset using global NV indices."""
    return [nv_list[int(ind)] for ind in inds]


def get_thresholds(nv_list):
    """Get fixed charge-state thresholds from nv_list."""
    thresholds = []
    for nv in nv_list:
        thresholds.append(float(nv.threshold))
    return np.array(thresholds)


def classify_counts_fixed_threshold(
    nv_list,
    counts_1d,
    confirm_margin_counts=0.0,
):
    """
    Classify each NV as NV- / not NV- using fixed thresholds.

    True  = classified as NV-
    False = classified as NV0 or uncertain

    A positive confirm_margin_counts makes the classification more conservative:
        counts > threshold + margin
    """
    thresholds = get_thresholds(nv_list)
    return counts_1d > (thresholds + confirm_margin_counts)


def extract_counts_1d(raw_data):
    """
    Convert base_routine raw counts into one mean count per active NV.

    Expected raw_data["counts"] structure is usually:
        counts[experiment][nv_ind, run_ind, step_ind, rep_ind]

    This function averages over all non-NV axes.
    """
    counts = np.array(raw_data["counts"])[0]
    num_nvs = counts.shape[0]
    counts_1d = counts.reshape(num_nvs, -1).mean(axis=1)
    return counts_1d


def set_readout_mask_for_active_nvs(
    active_global_inds,
    dmd=None,
    dmd_radius_px=4,
    dmd_plane=230,
    use_dmd=True,
):
    """
    Update DMD so that only active NVs are read out.

    Replace this with your actual DMD API calls.

    For your previous DMD code, something like this was used:
        dmd.pass_loaded_indices(json.dumps(active_global_inds), radius, plane)

    The main idea:
        - active NVs: allow readout light
        - confirmed NV- NVs: block readout light
    """

    active_global_inds = [int(ind) for ind in active_global_inds]

    if use_dmd:
        if dmd is None:
            dmd = tb.get_server_dmd()

        dmd.pass_loaded_indices(
            json.dumps(active_global_inds),
            dmd_radius_px,
            dmd_plane,
        )
        pass

    time.sleep(0.05)


# -----------------------------
# One active-set attempt
# -----------------------------

def run_one_active_charge_attempt(
    nv_list_all,
    active_global_inds,
    do_charge_polarize,
    num_reps=1,
    num_runs=1,
    seq_file="adaptive_charge_initialization.py",
    save_images=False,
):
    """
    Run one adaptive charge-init attempt on only the active NVs.

    active_global_inds:
        global indices of NVs still not confirmed NV-

    do_charge_polarize:
        False for initial readout-only check
        True for later attempts
    """

    active_nv_list = subset_nv_list(nv_list_all, active_global_inds)
    num_steps = 1

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        ion_coords_list = widefield.get_coords_list(
            active_nv_list, VirtualLaserKey.ION
        )

        pol_coords_list, pol_duration_list, pol_amp_list = (
            widefield.get_pulse_parameter_lists(
                active_nv_list, VirtualLaserKey.CHARGE_POL
            )
        )

        seq_args = [
            ion_coords_list,
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            bool(do_charge_polarize),
        ]

        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(seq_file, seq_args_string, num_reps)

    raw_data = base_routine.main(
        active_nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        save_images=save_images,
        save_images_avg_reps=False,
        charge_prep_fn=None,
        num_exps=1,
    )

    return raw_data


# -----------------------------
# Main adaptive routine
# -----------------------------

def adaptive_charge_initialize_to_nvm(
    nv_list,
    max_attempts=6,
    initial_readout=True,
    confirm_margin_counts=0.0,
    num_reps_per_attempt=1,
    num_runs_per_attempt=1,
    dmd_radius_px=4,
    dmd_plane=230,
    use_dmd=True,
    save_images=False,
):
    """
    Adaptive NV- initialization with active-set feedback.

    Parameters
    ----------
    nv_list : list
        Full NV list.

    max_attempts : int
        Maximum number of charge-polarization attempts after the initial readout.

    initial_readout : bool
        If True, first check which NVs are already NV- before applying charge prep.

    confirm_margin_counts : float
        Extra counts above threshold required to confirm NV-.
        Useful because false-positive confirmation is bad: once confirmed, the NV is
        removed from future prep/readout.

    num_reps_per_attempt : int
        For real initialization, keep this small, usually 1.

    num_runs_per_attempt : int
        For real initialization, keep this small, usually 1.

    dmd_radius_px : int
        DMD aperture radius for active readout mask.

    dmd_plane : int
        DMD plane parameter used by your DMD server.

    Returns
    -------
    final_data : dict
        Summary of adaptive initialization.
    """

    num_nvs = len(nv_list)

    confirmed_nvm = np.zeros(num_nvs, dtype=bool)
    attempt_confirmed = np.full(num_nvs, fill_value=-1, dtype=int)

    counts_history = np.full((num_nvs, max_attempts + 1), np.nan)
    state_history = np.full((num_nvs, max_attempts + 1), False, dtype=bool)
    active_history = []
    attempt_raw_data = []

    # Attempt index:
    #   0 can be readout-only
    #   1...max_attempts are charge-polarize + readout
    total_attempts = max_attempts + 1 if initial_readout else max_attempts

    for attempt_ind in range(total_attempts):
        active_global_inds = np.flatnonzero(~confirmed_nvm)
        active_history.append(active_global_inds.copy())

        print("\n----------------------------------------")
        print(f"Adaptive charge-init attempt {attempt_ind}")
        print(f"Active NVs remaining: {len(active_global_inds)} / {num_nvs}")

        if len(active_global_inds) == 0:
            print("All NVs confirmed NV-.")
            break

        # Initial attempt can be readout-only.
        if initial_readout and attempt_ind == 0:
            do_charge_polarize = False
            print("Mode: initial readout only")
        else:
            do_charge_polarize = True
            print("Mode: charge-polarize active NVs, then readout active NVs")

        # Update DMD readout mask so only active NVs are exposed/read out.
        set_readout_mask_for_active_nvs(
            active_global_inds,
            dmd_radius_px=dmd_radius_px,
            dmd_plane=dmd_plane,
            use_dmd=use_dmd,
        )

        # Run one active-set attempt.
        raw_attempt = run_one_active_charge_attempt(
            nv_list,
            active_global_inds,
            do_charge_polarize=do_charge_polarize,
            num_reps=num_reps_per_attempt,
            num_runs=num_runs_per_attempt,
            save_images=save_images,
        )

        attempt_raw_data.append(raw_attempt)

        # Extract and classify counts.
        active_nv_list = subset_nv_list(nv_list, active_global_inds)
        counts_1d = extract_counts_1d(raw_attempt)

        states_1d = classify_counts_fixed_threshold(
            active_nv_list,
            counts_1d,
            confirm_margin_counts=confirm_margin_counts,
        )

        counts_history[active_global_inds, attempt_ind] = counts_1d
        state_history[active_global_inds, attempt_ind] = states_1d

        newly_confirmed = active_global_inds[states_1d]
        confirmed_nvm[newly_confirmed] = True
        attempt_confirmed[newly_confirmed] = attempt_ind

        print(f"Newly confirmed NV-: {len(newly_confirmed)}")
        print(f"Total confirmed NV-: {np.sum(confirmed_nvm)} / {num_nvs}")
        print(f"Fraction confirmed NV-: {np.mean(confirmed_nvm):.3f}")

    # After experiment, block all readout or reset DMD safely.
    try:
        dmd = tb.get_server_dmd()
        dmd.pass_loaded_indices(json.dumps([]), dmd_radius_px, dmd_plane)
    except Exception:
        pass

    timestamp = dm.get_time_stamp()
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    final_data = {
        "timestamp": timestamp,
        "nv_list": nv_list,
        "num_nvs": num_nvs,
        "max_attempts": max_attempts,
        "initial_readout": initial_readout,
        "confirm_margin_counts": confirm_margin_counts,
        "num_reps_per_attempt": num_reps_per_attempt,
        "num_runs_per_attempt": num_runs_per_attempt,
        "dmd_radius_px": dmd_radius_px,
        "dmd_plane": dmd_plane,
        "confirmed_nvm": confirmed_nvm,
        "attempt_confirmed": attempt_confirmed,
        "counts_history": counts_history,
        "state_history": state_history,
        "active_history": active_history,
        "final_fraction_nvm": float(np.mean(confirmed_nvm)),
        "attempt_raw_data": attempt_raw_data,
        "notes": (
            "Adaptive active-set charge initialization. Confirmed NV- sites are "
            "removed from later charge-polarization and readout masks to reduce "
            "destructive readout."
        ),
    }

    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(final_data, file_path)

    try:
        fig = plot_adaptive_charge_init(final_data)
        dm.save_figure(fig, file_path)
    except Exception:
        print(traceback.format_exc())

    tb.reset_cfm()

    return final_data


# -----------------------------
# Plotting
# -----------------------------

def plot_adaptive_charge_init(raw_data):
    confirmed_nvm = np.array(raw_data["confirmed_nvm"])
    attempt_confirmed = np.array(raw_data["attempt_confirmed"])
    counts_history = np.array(raw_data["counts_history"])
    max_attempts = raw_data["max_attempts"]
    initial_readout = raw_data["initial_readout"]

    num_nvs = len(confirmed_nvm)
    total_attempts = max_attempts + 1 if initial_readout else max_attempts

    fraction_confirmed = []
    active_remaining = []

    for attempt_ind in range(total_attempts):
        confirmed_by_attempt = (attempt_confirmed >= 0) & (
            attempt_confirmed <= attempt_ind
        )
        fraction_confirmed.append(np.sum(confirmed_by_attempt) / num_nvs)
        active_remaining.append(num_nvs - np.sum(confirmed_by_attempt))

    fig, ax = plt.subplots()
    ax.plot(range(total_attempts), fraction_confirmed, marker="o")
    ax.set_xlabel("Adaptive attempt index")
    ax.set_ylabel("Fraction confirmed NV$^{-}$")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(range(total_attempts), active_remaining, marker="s", linestyle="--")
    ax2.set_ylabel("Active NVs remaining")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Example:
    # data = dm.get_raw_data(file_id=YOUR_NV_LIST_FILE_ID)
    # nv_list = data["nv_list"]
    #
    # adaptive_charge_initialize_to_nvm(
    #     nv_list,
    #     max_attempts=6,
    #     initial_readout=True,
    #     confirm_margin_counts=0.0,
    #     num_reps_per_attempt=1,
    #     num_runs_per_attempt=1,
    #     dmd_radius_px=4,
    #     dmd_plane=230,
    #     use_dmd=True,
    #     save_images=False,
    # )
    pass