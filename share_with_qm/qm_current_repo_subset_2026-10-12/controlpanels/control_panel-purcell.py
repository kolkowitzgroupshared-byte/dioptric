# -*- coding: utf-8 -*-
"""
QM review subset of the current Purcell control panel.

Source: controlpanels/control_panel-purcell.py
Source commit: bc290d8da6e529cdea0679bda9470e54b05fb8f0

This file is intentionally curated for the included QM-facing workflows.
The retained function bodies and the NVSig/NV-list construction block are
copied directly from the current NVOffice control panel. Unrelated experiment
launchers and analysis helpers are omitted.
"""

import numpy as np

from majorroutines.widefield import (
    adaptive_charge_initialization,
    charge_state_conditional_init,
    image_sample,
    spin_echo,
)
from utils import common, widefield
from utils import data_manager as dm
from utils import positioning as pos
from utils.constants import CoordsKey, NVSig, VirtualLaserKey

green_laser = "laser_INTE_520"
red_laser = "laser_COBO_638"
yellow_laser = "laser_OPTO_589"
green_laser_aod = f"{green_laser}_aod"
red_laser_aod = f"{red_laser}_aod"

# Relevant experiment entry points from the production control panel.
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
    # revival_period = int(35.66e3 / 2)  # ~52.39 G
    revival_period = int(55.77e3 / 2)  # ~33.39 G
    min_tau = 200
    taus = []
    revival_width = 6e3
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
    num_reps = 3
    num_runs = 600
    
    for _ in range(3):
        do_widefield_image_sample(nv_sig, 50)
        spin_echo.main(nv_list, num_steps, num_reps, num_runs, taus=taus)

# Relevant NV-coordinate / threshold helpers from the production panel.
def load_nv_coords(
    file_path="slmsuite/nv_blob_detection/nv_blob_filtered_multiple_nv302.npz",
    x_min=0,
    x_max=450,
    y_min=0,
    y_max=450,
    edge_tol=1.0,
):
    with np.load(file_path, allow_pickle=True) as data:
        nv_coordinates = np.asarray(data["nv_coordinates"], dtype=np.float32)

    mask = (
        (nv_coordinates[:, 0] >= x_min - edge_tol)
        & (nv_coordinates[:, 0] <= x_max + edge_tol)
        & (nv_coordinates[:, 1] >= y_min - edge_tol)
        & (nv_coordinates[:, 1] <= y_max + edge_tol)
    )

    nv_coordinates_clean = nv_coordinates[mask]

    print("Original NVs:", len(nv_coordinates))
    print("After x/y filter:", len(nv_coordinates_clean))

    dropped_inds = np.where(~mask)[0]
    print("Dropped NVs:", len(dropped_inds))

    if len(dropped_inds) > 0:
        print("Dropped indices:", dropped_inds.tolist())
        print("Dropped coordinates:")
        print(nv_coordinates[dropped_inds])

    return nv_coordinates_clean

def load_thresholds(file_path="slmsuite/nv_blob_detection/threshold_list_nvs_162.npz"):
    with np.load(file_path) as data_file:
        thresholds = data_file["arr_0"]
    return thresholds

# Current production NVSig / nv_list construction.
if __name__ == "__main__":
    # region Shared parameters
    green_coords_key = f"coords-{green_laser}"
    red_coords_key = f"coords-{red_laser}"
    pixel_coords_key = "pixel_coords"
    sample_name = "johnson"
    magnet_angle = 90
    date_str = "2026_10_02"
    sample_coords = [0.1, 0.0]
    z_coord = -0.1    
    # z_coord = 0.4

    
    config = common.get_config_dict()
    file_path = config["SpatialCalibrations"]["active_nv_coords_path"]
    pixel_coords_list = load_nv_coords(file_path=file_path).tolist()
    green_coords_list = [
        [
            round(coord, 3)
            for coord in pos.transform_coords(
                nv_pixel_coords, CoordsKey.PIXEL, green_laser_aod
            )
        ]
        for nv_pixel_coords in pixel_coords_list
    ]

    red_coords_list = [
        [
            round(coord, 3)
            for coord in pos.transform_coords(
                nv_pixel_coords, CoordsKey.PIXEL, red_laser_aod
            )
        ]
        for nv_pixel_coords in pixel_coords_list
    ]
    
    # -------------------------------------------
    # num_nvs
    # -------------------------------------------
    num_nvs = len(pixel_coords_list)
    
    # -------------------------------------------
    # Thresholds
    # -------------------------------------------

    threshold_list = [None] * num_nvs

    # -------------------------------------------
    # amplitudes
    # -------------------------------------------
    charge_pol_amps = [
        round(
            widefield.green_qua_amp_fn_2d(green_coords_list[i]),
            4,
        )
        for i in range(num_nvs)
    ]

    scc_amp_list = [
        round(
            widefield.red_qua_amp_fn_2d(red_coords_list[i]),
            4,
        )
        for i in range(num_nvs)
    ]
            
    # -------------------------------------------
    # durations
    # -------------------------------------------
    scc_dur_data = dm.get_raw_data(
        file_stem="2026_09_16-10_48_36-scc_parameter_sweep_analysis_with_nv_amps",
        load_npz=True,
    )
    scc_dur_dict = scc_dur_data["optimal_value_by_nv"]

    # Full list for all NVs
    scc_duration_list =[
        round(scc_dur_dict.get(i, scc_dur_dict.get(str(i))), 4)
        for i in range(len(scc_dur_dict))
    ]

    
    pol_duration_list = [1000] * num_nvs
    ion_duration_list = [1000] * num_nvs
    scc_duration_list = [88] * num_nvs

    # nv_list[i] will have the ith coordinates from the above lists
    nv_list: list[NVSig] = []
    for ind in range(num_nvs):
        # if ind not in selected_inds:
        #     continue
        coords = {
            CoordsKey.SAMPLE: sample_coords,
            CoordsKey.Z: z_coord,
            CoordsKey.PIXEL: pixel_coords_list[ind],
            green_laser_aod: green_coords_list[ind],    
            red_laser_aod: red_coords_list[ind],
        }
        nv_sig = NVSig(
            name=f"{sample_name}-nv{ind}_{date_str}",
            coords=coords,
            threshold=threshold_list[ind],
            pulse_durations={
                VirtualLaserKey.CHARGE_POL: pol_duration_list[ind],
                VirtualLaserKey.ION: ion_duration_list[ind],
                VirtualLaserKey.SCC: scc_duration_list[ind], 
            },
            pulse_amps={
                VirtualLaserKey.CHARGE_POL: charge_pol_amps[ind],
                VirtualLaserKey.ION: scc_amp_list[ind],  
                VirtualLaserKey.SCC: scc_amp_list[ind],
            },
        )
        nv_list.append(nv_sig)
    # Additional properties for the representative NV
    nv_list[0].representative = True
    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig = widefield.get_repr_nv_sig(nv_list)
    nv_sig.expected_counts = 1207.0
    # nv_list = nv_list[::-1]  # flipping the order of NVs
    # nv_list = nv_list[:300]
    print(f"length of NVs list:{len(nv_list)}")


    # QM-review experiment launch points.
    # Intentionally commented so this review file cannot accidentally start hardware.
    # do_widefield_image_sample(nv_sig, 50)
    # do_charge_state_conditional_init(nv_list)
    # do_adaptive_charge_initialization(nv_list)
    # do_spin_echo(nv_list)
