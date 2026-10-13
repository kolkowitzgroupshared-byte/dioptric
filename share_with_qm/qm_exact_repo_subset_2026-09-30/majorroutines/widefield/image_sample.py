# -*- coding: utf-8 -*-
"""
Illuminate an area, collecting onto the camera.

Created: April 9, 2019 (M. Cambria)
Updated: Fall 2024 (Saroj Chand)
"""


import copy
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
from majorroutines.targeting import optimize_pixel
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import NVSig, VirtualLaserKey


def single_nv(nv_sig, num_reps=1):
    nv_list = [nv_sig]
    return _nv_list_sub(nv_list, "single_nv", num_reps=num_reps)


def single_nv_ionization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_ionization"
    return _charge_state_prep_diff(nv_sig, caller_fn_name, num_reps)


def single_nv_polarization(nv_sig, num_reps=1):
    caller_fn_name = "single_nv_polarization"
    return _charge_state_prep_diff(nv_sig, caller_fn_name, num_reps)


def _charge_state_prep_diff(nv_sig, caller_fn_name, num_reps=1):
    if caller_fn_name == "single_nv_polarization":
        do_polarize_sig = True
        do_polarize_ref = False
        do_ionize_sig = False
        do_ionize_ref = False
    elif caller_fn_name == "single_nv_ionization":
        do_polarize_sig = True
        do_polarize_ref = True
        do_ionize_sig = True
        do_ionize_ref = False

    # Do the experiments
    sig_img_array = _charge_state_prep(
        nv_sig,
        caller_fn_name,
        num_reps,
        do_polarize=do_polarize_sig,
        do_ionize=do_ionize_sig,
    )
    ref_img_array = _charge_state_prep(
        nv_sig,
        caller_fn_name,
        num_reps,
        do_polarize=do_polarize_ref,
        do_ionize=do_ionize_ref,
    )

    # Calculate the difference and save
    fig, ax = plt.subplots()
    diff = sig_img_array - ref_img_array
    kpl.imshow(ax, diff, title="Difference", cbar_label="ADUs")
    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, timestamp, nv_sig["name"])
    dm.save_figure(fig, file_path)

    ### Get the pixel values of the NV in both images and a background level

    bg_offset = [10, -10]
    img_arrays = [sig_img_array, ref_img_array]
    titles = ["Signal", "Reference"]

    for ind in range(2):
        img_array = img_arrays[ind]
        title = titles[ind]

        # nv_pixel_coords = optimize_pixel(
        #     img_array,
        #     nv_sig,
        #     set_scanning_drift=False,
        #     set_pixel_drift=False,
        #     pixel_drift_adjust=False,
        # )
        # nv_counts = widefield.counts_from_img_array(
        #     img_array, nv_pixel_coords, drift_adjust=False
        # )
        nv_pixel_coords, nv_counts = optimize_pixel(nv_sig, img_array)

        bg_pixel_coords = [
            nv_pixel_coords[0] + bg_offset[0],
            nv_pixel_coords[1] + bg_offset[1],
        ]
        bg_counts = widefield.counts_from_img_array(
            img_array, bg_pixel_coords, drift_adjust=False
        )

        print(title)
        print(f"nv_counts: {nv_counts}")
        print(f"bg_counts: {bg_counts}")
        print(f"diff: {nv_counts - bg_counts}")
        print()


def _charge_state_prep(
    nv_sig,
    caller_fn_name,
    num_reps=1,
    save_dict=None,
    do_polarize=False,
    do_ionize=False,
):
    return main(
        nv_sig,
        caller_fn_name,
        num_reps=num_reps,
        save_dict=save_dict,
        do_polarize=do_polarize,
        do_ionize=do_ionize,
    )


def nv_list(nv_list, num_reps=1):
    save_dict = {"nv_list": nv_list}
    return _nv_list_sub(nv_list, "nv_list", save_dict, num_reps)


def _nv_list_sub(nv_list, caller_fn_name, save_dict=None, num_reps=1):
    nv_sig = nv_list[0]
    laser_key = VirtualLaserKey.IMAGING
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    laser_name = laser_dict["name"]
    adj_coords_list = [pos.get_nv_coords(nv, laser_name) for nv in nv_list]
    x_coords = [coords[0] for coords in adj_coords_list]
    y_coords = [coords[1] for coords in adj_coords_list]
    return main(nv_sig, caller_fn_name, num_reps, x_coords, y_coords, save_dict)


def widefield_image(nv_sig, num_reps=1):
    return main(nv_sig, "widefield", num_reps)


def widefield_scanning(nv_sig, x_range, y_range, num_steps):
    laser_key = VirtualLaserKey.WIDEFIELD_IMAGING
    laser_dict = tb.get_virtual_laser_dict(laser_key)
    laser_name = laser_dict["name"]
    x_center, y_center = [0, 0]
    ret_vals = pos.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, num_steps, num_steps
    )
    x_coords, y_coords, x_coords_1d, y_coords_1d, _ = ret_vals
    x_coords = list(x_coords)
    y_coords = list(y_coords)
    save_dict = {
        "x_range": x_range,
        "y_range": y_range,
        "x_coords_1d": x_coords_1d,
        "y_coords_1d": y_coords_1d,
    }
    num_reps = 1
    return main(nv_sig, "scanning", num_reps, x_coords, y_coords, save_dict)


def scanning(nv_sig, x_range, y_range, num_steps):
    laser_key = VirtualLaserKey.IMAGING
    positioner = pos.get_laser_positioner(laser_key)
    center_coords = pos.get_nv_coords(nv_sig, positioner)
    x_center, y_center = center_coords[0:2]
    ret_vals = pos.get_scan_grid_2d(
        x_center, y_center, x_range, y_range, num_steps, num_steps
    )
    x_coords, y_coords, x_coords_1d, y_coords_1d, _ = ret_vals
    x_coords = list(x_coords)
    y_coords = list(y_coords)
    save_dict = {
        "x_range": x_range,
        "y_range": y_range,
        "x_coords_1d": x_coords_1d,
        "y_coords_1d": y_coords_1d,
    }
    num_reps = 1
    return main(nv_sig, "scanning", num_reps, x_coords, y_coords, save_dict)


def scanning_full_roi(nv_sig, total_range, scan_range, num_steps):
    """
    Perform a full ROI scan by scanning across a frequency grid while iterating over spatial positions.

    Parameters:
        nv_sig: The NV signature to scan.
        total_range: The full frequency range to scan (e.g., ±15 MHz).
        scan_range: The scan window per step.
        num_steps: Number of spatial steps per scan.
    """
    # Get center coordinates
    laser_key = VirtualLaserKey.IMAGING
    positioner = pos.get_laser_positioner(laser_key)
    center_coords = pos.get_nv_coords(nv_sig, positioner)
    print(f"Center coordinates: {center_coords}")

    half_total_range = total_range / 2
    scan_frequencies = np.linspace(-half_total_range, half_total_range, num_steps)

    all_scan_data = []
    save_dict = {
        "scan_frequencies": scan_frequencies.tolist(),
        "total_range": total_range,
        "scan_range": scan_range,
        "num_steps": num_steps,
    }

    # Define scan grid
    steps = int(np.ceil(total_range / scan_range))
    x_offsets = np.linspace(-half_total_range, half_total_range, steps + 3)
    y_offsets = np.linspace(-half_total_range, half_total_range, steps + 3)

    for x_offset in x_offsets:
        for y_offset in y_offsets:
            # Create a deep copy of nv_sig to prevent overwriting original data
            updated_nv_sig = copy.deepcopy(nv_sig)

            # Update green laser coordinates (ensure correct key usage)
            updated_nv_sig.coords["laser_INTE_520_aod"] = [
                center_coords[0] + x_offset,
                center_coords[1] + y_offset,
            ]

            # Perform scanning at this position
            scan_data = scanning(updated_nv_sig, scan_range, scan_range, num_steps)
            all_scan_data.append(
                {"x_offset": x_offset, "y_offset": y_offset, "scan_data": scan_data}
            )

    # Save scan data
    save_dict["scanned_data"] = all_scan_data
    timestamp = dm.get_time_stamp()
    nv_name = nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_raw_data(save_dict, file_path, keys_to_compress=["scanned_data"])

    return file_path


# def scanning_full_roi(nv_sig, total_range, scan_range, num_steps):
#     """
#     Perform a full ROI scan by scanning across a frequency grid while iterating over spatial positions.

#     Parameters:
#         nv_sig: The NV signature to scan.
#         total_range: The full frequency range to scan (e.g., ±15 MHz).
#         scan_range: The scan window per step.
#         num_steps: Number of spatial steps per scan.
#     """
#     # Get center coordinates
#     laser_key = VirtualLaserKey.IMAGING
#     positioner = pos.get_laser_positioner(laser_key)
#     center_coords = pos.get_nv_coords(nv_sig, positioner)
#     print(f"Center coordinates: {center_coords}")

#     half_total_range = total_range / 2
#     scan_frequencies = np.linspace(-half_total_range, half_total_range, num_steps)

#     all_scan_data = []
#     save_dict = {
#         "scan_frequencies": scan_frequencies.tolist(),
#         "total_range": total_range,
#         "scan_range": scan_range,
#         "num_steps": num_steps,
#     }

#     # Define scan grid (unchanged)
#     steps = int(np.ceil(total_range / scan_range))
#     x_offsets = np.linspace(-half_total_range, half_total_range, steps + 1)
#     y_offsets = np.linspace(-half_total_range, half_total_range, steps + 1)

#     # --- NEW: two passes — original, then shifted by 0.8*scan_range ---
#     shift = 0.8 * scan_range
#     offset_sets = [
#         (x_offsets, y_offsets),  # original
#         (
#             np.clip(x_offsets + shift, -half_total_range, half_total_range),
#             np.clip(y_offsets + shift, -half_total_range, half_total_range),
#         ),  # shifted
#     ]

#     for xo, yo in offset_sets:
#         for x_offset in xo:
#             for y_offset in yo:
#                 # Create a deep copy of nv_sig to prevent overwriting original data
#                 updated_nv_sig = copy.deepcopy(nv_sig)

#                 # Update green laser coordinates (ensure correct key usage)
#                 updated_nv_sig.coords["laser_INTE_520_aod"] = [
#                     center_coords[0] + x_offset,
#                     center_coords[1] + y_offset,
#                 ]

#                 # Perform scanning at this position
#                 scan_data = scanning(updated_nv_sig, scan_range, scan_range, num_steps)
#                 all_scan_data.append(
#                     {
#                         "x_offset": float(x_offset),
#                         "y_offset": float(y_offset),
#                         "scan_data": scan_data,
#                     }
#                 )

#     # Save scan data
#     save_dict["scanned_data"] = all_scan_data
#     timestamp = dm.get_time_stamp()
#     nv_name = nv_sig.name
#     file_path = dm.get_file_path(__file__, timestamp, nv_name)
#     dm.save_raw_data(save_dict, file_path, keys_to_compress=["scanned_data"])

#     return file_path


def red_widefield_calibration(
    nv_sig,
    x_freqs_MHz,
    y_freqs_MHz,
    force_laser_key=None,
    num_reps=1,
):
    """
    Single camera exposure while stepping exactly 3 AOD (x,y) frequency pairs.
    Reuses simple_readout-scanning.py so the camera stays ON across the loop.
    """

    save_dict = {
        "mode": "calibration",
        "x_freqs_MHz": list(x_freqs_MHz),
        "y_freqs_MHz": list(y_freqs_MHz),
    }

    return main(
        nv_sig,
        caller_fn_name="calibration",
        num_reps=num_reps,
        x_coords=list(x_freqs_MHz),
        y_coords=list(y_freqs_MHz),
        save_dict=save_dict,
        force_laser_key=force_laser_key,
    )


def main(
    nv_sig: NVSig,
    caller_fn_name,
    num_reps=1,
    x_coords=None,
    y_coords=None,
    save_dict=None,
    do_polarize=False,
    do_ionize=False,
    force_laser_key=None,  # SBC <— NEW for red calibration
):
    ### Some initial setup

    tb.reset_cfm()
    # Choose virtual laser: widefield vs imaging, unless overridden
    if force_laser_key is not None:
        laser_key = force_laser_key
    else:
        laser_key = (
            VirtualLaserKey.WIDEFIELD_IMAGING
            if caller_fn_name == "widefield"
            else VirtualLaserKey.IMAGING
        )
    # targeting.pos.set_xyz_on_nv(nv_sig)
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    laser_dict = tb.get_virtual_laser_dict(laser_key)
    readout_laser = tb.get_physical_laser_name(laser_key)

    pos.set_xyz_on_nv(nv_sig)

    ### Load the pulse generator

    readout = laser_dict["duration"]
    readout_ms = readout / 10**6

    if caller_fn_name in ["scanning", "nv_list", "single_nv"]:
        seq_args = [readout, readout_laser, list(x_coords), list(y_coords)]
        seq_file = "simple_readout-scanning.py"

    elif caller_fn_name == "widefield":
        seq_args = [readout, readout_laser]
        seq_file = "simple_readout-widefield.py"

    elif caller_fn_name in ["single_nv_ionization", "single_nv_polarization"]:
        nv_list = [nv_sig]
        pol_coords_list = widefield.get_pol_coords_list(nv_list)
        ion_coords_list = widefield.get_ion_coords_list(nv_list)
        seq_args = [pol_coords_list, ion_coords_list]
        raise RuntimeError(
            "The sequence simple_readout-charge_state_prep needs to be updated "
            "to match the format of the seq_args returned by get_base_scc_seq_args"
        )
        seq_args.extend([do_polarize, do_ionize])
        seq_file = "simple_readout-charge_state_prep.py"

    elif caller_fn_name == "calibration":
        # Uses the scanning sequence, but with exactly the 3 tones you pass in.
        # Camera stays high the whole time; each tone gets the standard readout duration.
        seq_args = [readout, readout_laser, list(x_coords), list(y_coords)]
        seq_file = "simple_readout-scanning.py"

    # print(seq_args)
    # print(seq_file)
    # return

    ### Set up the image display

    title = f"{caller_fn_name}, {readout_laser}, {readout_ms} ms"

    ### Collect the data

    img_array_list = []

    def rep_fn(rep_ind):
        img_str = camera.read()
        sub_img_array, baseline = widefield.img_str_to_array(img_str)
        img_array_list.append(sub_img_array)
        # print(baseline)

    seq_args_string = tb.encode_seq_args(seq_args)
    pulse_gen.stream_load(seq_file, seq_args_string, num_reps)
    camera.arm()
    widefield.rep_loop(num_reps, rep_fn)
    camera.disarm()

    img_array = np.mean(img_array_list, axis=0)
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, title=title, cbar_label="ADUs")

    ### Clean up and save the data

    tb.reset_cfm()

    plt.show()

    kpl.show()

    timestamp = dm.get_time_stamp()
    raw_data = {
        "timestamp": timestamp,
        "caller_fn_name": caller_fn_name,
        "nv_sig": nv_sig,
        "num_reps": num_reps,
        "readout": readout_ms,
        "readout-units": "ms",
        "title": title,
        "img_array": img_array,
        "img_array-units": "counts",
    }
    if save_dict is not None:
        raw_data |= save_dict  # Add in the passed info to save

    nv_name = nv_sig.name
    file_path = dm.get_file_path(__file__, timestamp, nv_name)
    dm.save_figure(fig, file_path)
    dm.save_raw_data(raw_data, file_path, keys_to_compress=["img_array"])

    return img_array


if __name__ == "__main__":
    kpl.init_kplotlib()

    # # Tweezered NVs
    # data = dm.get_raw_data(file_id=1693166192526, load_npz=True)
    # img_array = data["ref_img_array"]
    # fig, ax = plt.subplots()
    # kpl.imshow(ax, img_array, cbar_label="Photons")
    # ax.axis("off")
    # scale = 4 * (2.3 / 0.29714285714)
    # kpl.scale_bar(ax, scale, "4 µm", kpl.Loc.LOWER_RIGHT)
    # kpl.show(block=True)
    # sys.exit()

    # # Tweezer pattern
    # data = np.load("/home/mccambria/Downloads/captured_image_raw.npy")
    # data = np.rot90(
    #     data,
    # )
    # fig, ax = plt.subplots()
    # kpl.imshow(ax, data, no_cbar=True)
    # ax.axis("off")
    # kpl.show(block=True)
    # sys.exit()

    # Composite green
    file_names = [
        "2024_11_05-20_07_31-johnson-nv0_2024_03_12",
        "2024_11_05-20_07_54-johnson-nv0_2024_03_12",
        "2024_11_05-20_09_07-johnson-nv0_2024_03_12",
        "2024_11_05-20_09_39-johnson-nv0_2024_03_12",
        "2024_11_05-20_10_06-johnson-nv0_2024_03_12",
        "2024_11_05-20_10_41-johnson-nv0_2024_03_12",
        "2024_11_05-20_11_06-johnson-nv0_2024_03_12",
        "2024_11_05-20_06_45-johnson-nv0_2024_03_12",
    ]
    img_arrays = []
    for file_name in file_names:
        data = dm.get_raw_data(file_name, load_npz=True)
        img_array = np.array(data["img_array"])
        img_array -= 300
        img_array = img_array / np.median(img_array)
        img_arrays.append(img_array)
    img_arrays = np.array(img_arrays)
    img_array = np.max(img_arrays, axis=0)
    # img_array = widefield.adus_to_photons(img_array, em_gain=10)
    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array)
    ax.axis("off")
    # scale = 4 * (2.3 / 0.29714285714)
    # kpl.scale_bar(ax, scale, "4 µm", kpl.Loc.LOWER_RIGHT)
    kpl.show(block=True)
    sys.exit()

    ######

    ### Basic widefield
    # data = dm.get_raw_data(file_id=1556420881234, load_npz=True)
    data = dm.get_raw_data(file_id=1556655608661, load_npz=True)
    img_array = np.array(data["img_array"])
    # img_array = widefield.adus_to_photons(img_array)
    img_array_offset = [0, 0]

    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array)
    ax.axis("off")

    scale = 2 * widefield.get_camera_scale()
    kpl.scale_bar(ax, scale, "2 µm", kpl.Loc.UPPER_RIGHT)

    kpl.show(block=True)

    ### Smiley
    smiley_inds = list(range(6))

    # Histograms ref
    # file_id = 1556690958663
    # img_array_offset = [3, 5]
    # ion_inds = []
    # img_array_ind = 1

    # Everything ionized
    # file_id = 1556690958663
    # img_array_offset = [3, 5]
    # ion_inds = [0, 1, 2, 3, 4, 5]
    # img_array_ind = 0

    # Every other
    file_id = 1556723203013
    img_array_offset = [0, 0]
    ion_inds = [0, 2, 5]
    img_array_ind = 0

    # # Inverted every other
    # file_id = 1556745846534
    # img_array_offset = [0,0]
    # ion_inds = [1, 3, 4]
    # img_array_ind = 0

    # # Blinking
    # file_id = 1556830424482
    # img_array_offset = [0,0]
    # ion_inds = [4,5]
    # img_array_ind = 0

    # # Winking
    # file_id = None
    # img_array_offset = [0,0]
    # ion_inds = [5]
    # img_array_ind = 0

    data = dm.get_raw_data(file_id=file_id, load_npz=True, use_cache=False)
    img_arrays = np.array(data["img_arrays"])
    del data
    img_arrays = img_arrays[img_array_ind]
    img_array = np.mean(img_arrays, axis=(0, 1, 2))
    del img_arrays
    not_ion_inds = [ind for ind in smiley_inds if ind not in ion_inds]

    ######

    widefield.replace_dead_pixel(img_array)
    img_array = img_array[
        10 + img_array_offset[0] : 240 + img_array_offset[0],
        10 + img_array_offset[1] : 240 + img_array_offset[1],
    ]

    pixel_coords_list = [
        [131.144, 129.272],  #  Smiley
        [161.477, 105.335],  #  Smiley
        [135.139, 104.013],
        [110.023, 87.942],
        [144.169, 163.787],
        [173.93, 78.505],  #  Smiley
        [171.074, 49.877],  #  Smiley
        [170.501, 132.597],
        [137.025, 74.662],
        [58.628, 139.616],
        # Smiley additions
        # [150.34, 119.249],  # Too much crosstalk
        [61.277, 76.387],
        [85.384, 33.935],
    ]
    pixel_coords_list = [
        widefield.adjust_pixel_coords_for_drift(el, [10 - 10, 38 - 10])
        for el in pixel_coords_list
    ]

    # coords_list = [
    #     widefield.get_nv_pixel_coords(nv, drift_adjust=True, drift=(4, 10))
    #     for nv in nv_list
    # ]
    # widefield.integrate_counts(img_array, coords_list[1])
    # print(coords_list)
    # scale = widefield.get_camera_scale()
    # um_coords_list = [
    #     (round(el[0] / scale, 3), round(el[1] / scale, 3)) for el in coords_list
    # ]
    # print(um_coords_list)

    # img_array = widefield.adus_to_photons(img_array)

    # Clean up dead pixel by taking average of nearby pixels
    # dead_pixel = [142, 109]
    # dead_pixel_x = dead_pixel[1]
    # dead_pixel_y = dead_pixel[0]
    # img_array[dead_pixel_y, dead_pixel_x] = np.mean(
    #     img_array[
    #         dead_pixel_y - 1 : dead_pixel_y + 1 : 2,
    #         dead_pixel_x - 1 : dead_pixel_x + 1 : 2,
    #     ]
    # )

    fig, ax = plt.subplots()
    kpl.imshow(ax, img_array, cbar_label="Photons", vmin=0.05, vmax=0.45)
    ax.axis("off")

    scale = 2 * widefield.get_camera_scale()
    kpl.scale_bar(ax, scale, "2 µm", kpl.Loc.UPPER_RIGHT)
    # pixel_coords_list = [pixel_coords_list[ind] for ind in range(10)]
    # widefield.draw_circles_on_nvs(ax, pixel_coords_list=pixel_coords_list)
    pixel_coords_list = [pixel_coords_list[ind] for ind in [0, 1, 5, 6, 10, 11]]
    draw_coords_list = [pixel_coords_list[ind] for ind in ion_inds]
    widefield.draw_circles_on_nvs(
        ax,
        pixel_coords_list=draw_coords_list,
        color=kpl.KplColors.DARK_GRAY,
        no_legend=True,
        linestyle="solid",
    )
    draw_coords_list = [pixel_coords_list[ind] for ind in not_ion_inds]
    widefield.draw_circles_on_nvs(
        ax,
        pixel_coords_list=draw_coords_list,
        color=kpl.KplColors.DARK_GRAY,
        no_legend=True,
        linestyle="dashed",
    )

    plt.show(block=True)
