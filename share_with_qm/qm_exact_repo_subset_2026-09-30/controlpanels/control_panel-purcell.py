# -*- coding: utf-8 -*-
"""
QM review subset of control_panel-purcell.py.

Source commit: 6fbe4cc788374d00b2c934045685ca775493e436

This file is intentionally trimmed to the major-routine entry points included
in this review package. The function bodies below are copied verbatim from the
production control panel at the source commit.

Note: production do_spin_echo() references the control-panel global 'nv_sig'
for the pre-run widefield image. The large sample/NV-definition section that
defines that global is intentionally omitted from this external review subset.
"""

import numpy as np

from majorroutines.widefield import (
    adaptive_charge_initialization,
    charge_state_conditional_init,
    image_sample,
    spin_echo,
)


def do_widefield_image_sample(nv_sig, num_reps=1):
    return image_sample.widefield_image(nv_sig, num_reps)


def do_charge_state_conditional_init(nv_list):
    num_reps = 10
    num_runs = 4
    # num_runs = 400
    return charge_state_conditional_init.main(nv_list, num_reps, num_runs)


def do_adaptive_charge_initialization(nv_list):
    adaptive_charge_initialization.main(
        nv_list,
        num_reps=10,
        num_runs=4,
        mode="dmd_block_confirmed",
        # mode="dmd_all_on",
        dmd_indices=None,      # full nv_list, DMD index = nv_list index
        dmd_radius_px=8,
        dmd_plane=230,
        confirm_margin_counts=1.0,
        save_images=True,
        save_images_avg_reps=False,
        save_data=True,
        save_fig=True,
        save_image_frames=True,
        save_movie=True,
    )


def do_spin_echo(nv_list):
    # revival_period = int(51.5e3 / 2) ### ~37.0 G
    # revival_period = int(38.5e3 / 2)### 49.68 G
    # revival_period = int(28.6e3 / 2) ### 65.14G
    # revival_period = int(29.90e3 / 2) ### 62.14G
    # revival_period = int(31.2e3 / 2) ### 59.69G
    revival_period = int(35.66e3 / 2)  # ~52.39 G
    min_tau = 200
    taus = []
    revival_width = 6e3
    # revival_width = 4e3
    decay = np.linspace(min_tau, min_tau + revival_width, 6)
    taus.extend(decay.tolist())
    gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 8)
    taus.extend(gap[1:-1].tolist())
    first_revival = np.linspace(
        revival_period - revival_width, revival_period + revival_width, 65
    )
    taus.extend(first_revival.tolist())
    gap = np.linspace(
        revival_period + revival_width, 2 * revival_period - revival_width, 8
    )
    taus.extend(gap[1:-1].tolist())
    second_revival = np.linspace(
        2 * revival_period - revival_width, 2 * revival_period + revival_width, 11
    )
    taus.extend(second_revival.tolist())
    taus = [round(el / 4) * 4 for el in taus]

    # Remove duplicates and sort
    taus = sorted(set(taus))

    # Experiment settings
    num_steps = len(taus)

    # Automatic taus setup, linear spacing
    # min_tau = 200
    # max_tau = 84e3 + min_tau
    # num_steps = 29

    num_reps = 3
    num_runs = 600
    # num_runs = 2
    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, min_tau, max_tau)
    # spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)
    for ind in range(3):
        do_widefield_image_sample(nv_sig, 50)
        spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)


# def do_spin_echo(nv_list):
#     min_tau = 200  # ns
#     revival_period = int(20e3) ##20 gauss
#     taus = []
#     revival_width = 6e3
#     decay = np.linspace(min_tau, min_tau + revival_width, 6)
#     taus.extend(decay.tolist())
#     gap = np.linspace(min_tau + revival_width, revival_period - revival_width, 6)
#     taus.extend(gap[1:-1].tolist())
#     first_revival = np.linspace(
#         revival_period - revival_width, revival_period + revival_width, 61
#     )
#     taus.extend(first_revival.tolist())
#     # Round to clock-cycle-compatible units
#     taus = [round(el / 4) * 4 for el in taus]
#     # Remove duplicates and sort
#     taus = sorted(set(taus))
#     num_steps = len(taus)
#     num_reps = 3
#     num_runs = 600

#     print(
#         f"[Spin Echo] Running with {num_steps} τ values, revival_period={revival_period}"
#     )

#     for _ in range(1):
#         spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)
