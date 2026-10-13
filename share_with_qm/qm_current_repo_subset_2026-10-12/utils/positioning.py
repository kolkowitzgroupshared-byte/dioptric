# -*- coding: utf-8 -*-
"""Various functions for positioning microscope focus. Includes
functions for generating lists of coordinates used in scans

Created on Decemeber 1st, 2022

@author: mccambria
"""

# region Imports and constants

import time
from functools import cache

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import common
from utils import tool_belt as tb
from utils.constants import (
    CollectionMode,
    CoordsKey,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)

# endregion
# region Simple sets
"""
If a specific laser is not passed, then the set will just use the global
coords (nv_sig key "coords"). Otherwise we'll use the laser specific coords
(nv_sig key f"coords-{positioner}")
"""

def set_xyz(coords, positioner=CoordsKey.SAMPLE, drift_adjust=None, ramp=None):
    if drift_adjust is None:
        drift_adjust = should_drift_adjust(positioner)
    if drift_adjust:
        coords = adjust_coords_for_drift(coords, coords_key=positioner)

    if ramp is None:
        config = common.get_config_dict()
        ramp = config.get("set_xyz_ramp", False)

    if ramp:
        # return _set_xyz_ramp(coords)
        return _set_xyz_ramp(coords, positioner)

    else:
        return _set_xyz(coords, positioner)


def _set_xyz(coords, positioner):
    positioner_server = get_positioner_server(positioner)

    if positioner_server is None:
        return

    if positioner is CoordsKey.Z:
        positioner_server.write_z(coords)
    else:
        positioner_server.write_xy(coords[0], coords[1])


def _set_xyz_ramp(coords, positioner):
    """Not up to date: Step incrementally to this position from the current position"""

    config = common.get_config_dict()
    config_positioning = config["Positioning"]

    xy_dtype = config_positioning["xy_dtype"]
    z_dtype = config_positioning["z_dtype"]

    step_size_xy = config_positioning["xy_incremental_step_size"]
    step_size_z = config_positioning["z_incremental_step_size"]

    xy_delay = config_positioning["xy_delay"]
    z_delay = config_positioning["z_delay"]

    # Take whichever one is longer
    if xy_delay > z_delay:
        total_movement_delay = xy_delay
    else:
        total_movement_delay = z_delay

    xyz_server = get_positioner_server(positioner)
    pulse_gen = tb.get_server_pulse_gen()

    # if the movement type is int, just skip this and move to the desired position
    if xy_dtype is int or z_dtype is int:
        set_xyz(coords)
        return

    # Get current and final position
    current_x, current_y = xyz_server.read_xy()
    current_z = xyz_server.read_z()
    final_x, final_y, final_z = coords

    dx = final_x - current_x
    dy = final_y - current_y
    dz = final_z - current_z
    # print('dx: {}'.format(dx))
    # print('dy: {}'.format(dy))

    # If we are moving a distance smaller than the step size,
    # just set the coords, don't try to run a sequence

    if abs(dx) <= step_size_xy and abs(dy) <= step_size_xy and abs(dz) <= step_size_z:
        # print('just setting coords without ramp')
        set_xyz(coords)

    else:
        # Determine num of steps to get to final destination based on step size
        num_steps_x = np.ceil(abs(dx) / step_size_xy)
        num_steps_y = np.ceil(abs(dy) / step_size_xy)
        num_steps_z = np.ceil(abs(dz) / step_size_z)

        # Determine max steps for this move
        max_steps = int(max([num_steps_x, num_steps_y, num_steps_z]))

        # The delay between steps will be the total delay divided by the num of incr steps
        movement_delay = int(total_movement_delay / max_steps)

        x_points = [current_x]
        y_points = [current_y]
        z_points = [current_z]

        # set up the voltages to step thru. Once x, y, or z reach their final
        # value, just pass the final position for the remaining steps
        for n in range(max_steps):
            if n > num_steps_x - 1:
                x_points.append(final_x)
            else:
                move_x = (n + 1) * step_size_xy * dx / abs(dx)
                incr_x_val = move_x + current_x
                x_points.append(incr_x_val)

            if n > num_steps_y - 1:
                y_points.append(final_y)
            else:
                move_y = (n + 1) * step_size_xy * dy / abs(dy)
                incr_y_val = move_y + current_y
                y_points.append(incr_y_val)

            if n > num_steps_z - 1:
                z_points.append(final_z)
            else:
                move_z = (n + 1) * step_size_z * dz / abs(dz)
                incr_z_val = move_z + current_z
                z_points.append(incr_z_val)
        # Run a simple clock pulse repeatedly to move through votlages
        file_name = "simple_clock.py"
        seq_args = [movement_delay]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(file_name, seq_args_string)
        # period = ret_vals[0]
        # print(z_points)

        xyz_server.load_stream_xyz(x_points, y_points, z_points)
        pulse_gen.stream_load(file_name, seq_args_string)
        pulse_gen.stream_start(max_steps)

    # Force some delay before proceeding to account
    # for the effective write time, as well as settling time for movement
    time.sleep(total_movement_delay / 1e9)


def set_xyz_on_nv(nv_sig, positioner=None, drift_adjust=None):
    """Sets XYZ coordinates for the NV. If positioner is None, set all available
    positioners.
    """
    if positioner is None:
        config = common.get_config_dict()
        positioners = config["Positioning"]["Positioners"].keys()
        for el in positioners:
            set_xyz_on_nv(nv_sig, positioner=el, drift_adjust=drift_adjust)
    else:
        coords = get_nv_coords(nv_sig, positioner, drift_adjust)
        set_xyz(coords, positioner=positioner, drift_adjust=False)
        return coords


def get_nv_coords(
    nv_sig: NVSig, coords_key=CoordsKey.SAMPLE, drift_adjust=None, drift=None
):
    if drift_adjust is None:
        drift_adjust = should_drift_adjust(coords_key)

    coords_val = nv_sig.coords
    if isinstance(coords_val, dict):
        coords = coords_val[coords_key]
    else:
        coords = coords_val
    if drift_adjust:
        coords = adjust_coords_for_drift(
            coords=coords, drift=drift, coords_key=coords_key
        )
    return coords


# def should_drift_adjust(coords_key):
#     """Check whether or not we should adjust the coordinates associated with the
#     passed coords_key for drift. Assume that we compensate for drift by adjusting
#     the sample positioner if we can. Otherwise, we adjust the coordinates
#     associated with the optical paths
#     """
#     if coords_key in [CoordsKey.SAMPLE, CoordsKey.Z]:
#         return True
#     else:
#         return not has_sample_positioner()

def should_drift_adjust(coords_key):
    """Decide whether coords for this positioner should be drift-adjusted.

    Small tweak:
    - camera mode: adjust SAMPLE and Z
    - counter/confocal mode: adjust PIXEL
    """
    config = common.get_config_dict()
    collection_mode = config["collection_mode"]

    if collection_mode == CollectionMode.CAMERA:
        if coords_key == CoordsKey.SAMPLE:
            return True
        if coords_key == CoordsKey.Z:
            return True
        return False

    # confocal / counter mode
    if coords_key == CoordsKey.PIXEL:
        return True
    if coords_key == CoordsKey.Z:
        return True
    return False

def set_nv_coords(nv_sig, coords, coords_key=CoordsKey.SAMPLE):
    coords_val = nv_sig.coords
    if isinstance(coords_val, list):
        nv_sig.coords = coords
    if isinstance(coords_val, dict):
        nv_sig.coords[coords_key] = coords


def get_nv_pulse_duration(nv_sig: NVSig, vkey: VirtualLaserKey) -> int:
    """
    Return per-NV pulse duration (ns) for a virtual laser, mirroring get_nv_coords style.
    Resilient to Enum/string key mismatches; falls back to config default.
    """
    pulse_dur_val = nv_sig.pulse_durations
    if isinstance(pulse_dur_val, dict):
        
        pulse_duraions = pulse_dur_val[vkey]
    else:
        pulse_duraions = pulse_dur_val
    
    return pulse_duraions
# endregion


# region Getters


def get_laser_pos_mode(laser_name):
    config = common.get_config_dict()
    config_laser = config["Optics"][laser_name]
    if "pos_mode" in config_laser:
        return config_laser["pos_mode"]
    else:
        return None


def has_sample_positioner():
    config = common.get_config_dict()
    return CoordsKey.SAMPLE in config["Positioning"]["Positioners"]


def has_z_positioner():
    config = common.get_config_dict()
    return CoordsKey.Z in config["Positioning"]["Positioners"]


def get_laser_positioner(virtual_laser_key: VirtualLaserKey):
    virtual_laser_dict = tb.get_virtual_laser_dict(virtual_laser_key)
    physical_laser_name = virtual_laser_dict["physical_name"]
    physical_laser_dict = tb.get_physical_laser_dict(physical_laser_name)
    return physical_laser_dict["positioner"]


def get_positioner_server(positioner):
    physical_name = _get_positioner_attr(positioner, "physical_name")
    server = common.get_server_by_name(physical_name)
    return server


def get_positioner_control_mode(positioner):
    return _get_positioner_attr(positioner, "control_mode")


def get_positioner_units(positioner):
    return _get_positioner_attr(positioner, "units")


def get_positioner_optimize_range(positioner):
    return _get_positioner_attr(positioner, "optimize_range")


def _get_positioner_attr(positioner, key):
    config = common.get_config_dict()
    positioner_dict = config["Positioning"]["Positioners"][positioner]
    if key in positioner_dict:
        return positioner_dict[key]
    else:
        return None


def get_positioner_write_fn(positioner, axis_ind):
    """Return the write function for a given axis (0:x, 1:y, 2:z)"""
    if axis_ind in [0, 1]:
        server = get_positioner_server(positioner)
    elif axis_ind == 2:
        server = get_positioner_server(positioner)
    if server is None:
        return None

    if axis_ind == 0:
        write_fn = server.write_x
    if axis_ind == 1:
        write_fn = server.write_y
    if axis_ind == 2:
        write_fn = server.write_z

    return write_fn


def get_positioner_stream_fn(positioner, axis_ind):
    """Return the stream function for a given axis (0:x, 1:y, 2:z)"""
    control_mode = get_positioner_control_mode(positioner)
    if control_mode != PosControlMode.STREAM:
        return None

    if axis_ind in [0, 1]:
        server = get_positioner_server(positioner)
    elif axis_ind == 2:
        server = get_positioner_server(positioner)
    if server is None:
        return None

    if axis_ind == 0:
        stream_fn = server.load_stream_x
    if axis_ind == 1:
        stream_fn = server.load_stream_y
    if axis_ind == 2:
        stream_fn = server.load_stream_z
        # stream_fn = server.load_scan_z
    return stream_fn


# endregion

# region Drift
"""Implemented with a drift tracking global stored on the registry"""


@cache
def get_drift(coords_key=None):
    key = "DRIFT"
    drift = common.get_registry_entry(["State"], key)
    drift = np.array(drift)
    drift_xy_coords_key = get_drift_xy_coords_key()
    if coords_key is None:
        return drift
    elif coords_key is CoordsKey.Z:
        return drift[2]
    elif coords_key is drift_xy_coords_key:
        return drift[0:2]

    # Drift must be transformed
    drift = transform_drift(drift[0:2], coords_key)

    # For sample compensation, we have to move opposite the measured direction
    if coords_key is CoordsKey.SAMPLE:
        drift = [-1 * el for el in drift]
    return drift


def transform_drift(drift, dest_coords_key):
    # Get the drift stored in the registry and use the calibration in the config
    # to convert it from the space in which it was calculated (i.e. imaging laser
    # positioner coordinates) to the space of the passed coords_key (e.g. SCC
    # laser positioner coordinates)
    transformation_matrix = get_drift_transformation_matrix(dest_coords_key)
    return apply_affine_transformation(drift, transformation_matrix)


def transform_coords(source_coords, source_coords_key, dest_coords_key):
    transformation_matrix = get_coordinate_transformation_matrix(
        source_coords_key, dest_coords_key
    )
    return apply_affine_transformation(source_coords, transformation_matrix)


def apply_affine_transformation(source_coords, transformation_matrix):
    # MCC
    return np.dot(transformation_matrix, np.append(source_coords, 1))


@cache
def get_coordinate_transformation_matrix(source_coords_key, dest_coords_key):
    return _get_transformation_matrix(
        source_coords_key, dest_coords_key, relative=False
    )


@cache
def get_drift_xy_coords_key():
    """Determine what coordinate space we store the xy drift in.
    Z is always in sample positioner space if we have a sample positioner
    that can move in z. Otherwise there's no way to keep track of z drift.
    """
    config = common.get_config_dict()
    # optional explicit override:
    explicit = config["Positioning"].get("drift_xy_coords_key")
    if explicit is not None:
        return explicit
    collection_mode = config["collection_mode"]
    if collection_mode == CollectionMode.CAMERA:
        return CoordsKey.PIXEL
    if has_sample_positioner():
        return CoordsKey.SAMPLE
    positioner = get_laser_positioner(VirtualLaserKey.IMAGING)
    return positioner


@cache
def get_drift_transformation_matrix(dest_coords_key):
    source_coords_key = get_drift_xy_coords_key()
    return _get_transformation_matrix(source_coords_key, dest_coords_key, relative=True)


def _get_transformation_matrix(source_coords_key, dest_coords_key, relative):
    # MCC
    if source_coords_key is CoordsKey.PIXEL and dest_coords_key is CoordsKey.SAMPLE:
        config = common.get_config_dict()
        key = "pixel_to_sample_affine_transformation_matrix"
        transformation_matrix = np.array(config["Positioning"][key])
        if relative:
            transformation_matrix[:, 2] = [0, 0]
        return transformation_matrix

    nvs = _get_coordinate_calibration_nvs()

    source_coords_arr = []
    dest_coords_arr = []
    for ind in range(3):
        nv = nvs[ind]
        source_coords = get_nv_coords(nv, source_coords_key, drift_adjust=False)
        dest_coords = get_nv_coords(nv, dest_coords_key, drift_adjust=False)

        if relative:
            if ind == 0:
                ref_source_coords = source_coords
                ref_dest_coords = dest_coords

            source_coords_diff = np.array(source_coords) - np.array(ref_source_coords)
            source_coords_arr.append(source_coords_diff)
            dest_coords_diff = np.array(dest_coords) - np.array(ref_dest_coords)
            dest_coords_arr.append(dest_coords_diff)
        else:
            source_coords_arr.append(source_coords)
            dest_coords_arr.append(dest_coords)

    source_coords_arr = np.array(source_coords_arr, dtype="float32")
    dest_coords_arr = np.array(dest_coords_arr, dtype="float32")
    transformation_matrix = cv2.getAffineTransform(source_coords_arr, dest_coords_arr)

    return transformation_matrix


def _get_coordinate_calibration_nvs():
    module = common.get_config_module()
    nv1 = NVSig(coords=module.calibration_coords_nv1)
    nv2 = NVSig(coords=module.calibration_coords_nv2)
    nv3 = NVSig(coords=module.calibration_coords_nv3)
    return nv1, nv2, nv3


def set_drift_val(drift_val, axis_ind, cumulative=False):
    drift = get_drift()
    get_drift.cache_clear()

    key = "DRIFT"
    if cumulative:
        drift[axis_ind] += drift_val
    else:
        drift[axis_ind] = drift_val
    return common.set_registry_entry(["State"], key, drift)


def set_drift(drift):
    get_drift.cache_clear()
    key = "DRIFT"
    return common.set_registry_entry(["State"], key, drift)


def reset_drift():
    try:
        drift = get_drift()
        len_drift = len(drift)
    except Exception:
        len_drift = 3
    return set_drift([0.0] * len_drift)


def reset_xy_drift():
    drift = get_drift()
    if len(drift) == 2:
        return set_drift([0.0, 0.0])
    else:
        return set_drift([0.0, 0.0, drift[2]])


def adjust_coords_for_drift(
    coords=None, drift=None, nv_sig=None, coords_key=CoordsKey.SAMPLE
):
    """Current drift will be retrieved from registry if passed drift is None"""
    if coords is None:
        coords = get_nv_coords(nv_sig, coords_key, drift_adjust=False)
    if drift is None:
        drift = get_drift(coords_key)

    scalar_coords = not hasattr(coords, "__len__")
    if scalar_coords:
        return coords + drift

    adjusted_coords = []
    for ind in range(len(coords)):
        coords_val = coords[ind]
        drift_val = drift[ind]
        if coords_val is not None and drift_val is not None:
            adj_val = coords_val + drift_val
        else:
            adj_val = None
        adjusted_coords.append(adj_val)
    return adjusted_coords


# endregion
# region Scans
"""These are really just calculator functions for generating lists of coordinates
used in scans. Since the functions don't care about whether your scan is xy or xz
or whatever, variables are named axis-agnostically as <var>_<axis_ind>
"""


def get_scan_1d(center, scan_range, num_steps, dtype=np.float64) -> np.ndarray:
    """Get a linear spacing of coords about the passed center

    Parameters
    ----------
    center : numeric
        Center of the scan along the axis
    range : numeric
        Full range of the scan along the first axis
    num_steps : _type_
        Number of steps in the scan

    Returns
    -------
    array(numeric)
        Scan coords
    """
    # print(f"Center: {center}, Scan Range: {scan_range}, Num Steps: {num_steps}")
    # if center is None or scan_range is None:
    #     raise ValueError("Center or Scan Range is None")

    half_range = scan_range / 2
    low = center - half_range
    high = center + half_range
    coords = np.linspace(low, high, num_steps, dtype=dtype)
    return coords


# load_sweep_scan_xy
def get_scan_grid_2d(
    center_1,
    center_2,
    scan_range_1,
    scan_range_2,
    num_steps_1,
    num_steps_2,
    dtype=np.float64,
):
    """Create a grid of points for a snake scan

    Parameters
    ----------
    center_1 : numeric
        Center of the scan along the first axis
    center_2 : numeric
        Center of the scan along the second axis
    scan_range_1 : numeric
        Full range of the scan along the first axis
    scan_range_2 : numeric
        Full range of the scan along the second axis
    num_steps_1 : int
        Number of steps along the first axis
    num_steps_2 : int
        Number of steps along the second axis

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the snake scan
        e.g.
    array(numeric)
        Values to write to the second axis for the snake scan
    array(numeric)
        First-axis coordinates (i.e. coordinates scanned through if second axis were fixed)
    array(numeric)
        Second-axis coordinates (i.e. coordinates scanned through if first axis were fixed)
    list(float)
        Extent of the grid in the form [left, right, bottom, top] - includes half-pixel adjusment to
        min/max written vals for each axis so that the pixels in an image are properly centered
    """

    coords_1_1d = get_scan_1d(center_1, scan_range_1, num_steps_1, dtype)
    coords_2_1d = get_scan_1d(center_2, scan_range_2, num_steps_2, dtype)

    ### Winding cartesian product
    # The first axis values are repeated - the second axis values are mirrored and tiled
    # The comments below shows what happens for [1, 2, 3], [4, 5, 6]

    # [1, 2, 3] => [1, 2, 3, 3, 2, 1]
    inter_1 = np.concatenate((coords_1_1d, np.flipud(coords_1_1d)))
    # [1, 2, 3, 3, 2, 1] => [1, 2, 3, 3, 2, 1, 1, 2, 3]
    if num_steps_2 % 2 == 0:
        coords_1 = np.tile(inter_1, num_steps_2 // 2)
    else:  # Odd x size
        coords_1 = np.tile(inter_1, num_steps_2 // 2)
        coords_1 = np.concatenate((coords_1, coords_1_1d))

    # [4, 5, 6] => [4, 4, 4, 5, 5, 5, 6, 6, 6]
    coords_2 = np.repeat(coords_2_1d, num_steps_1)

    x_low = min(coords_1_1d)
    x_high = max(coords_1_1d)
    y_low = min(coords_2_1d)
    y_high = max(coords_2_1d)

    # For the image extent, we need to bump out the min/max axis coords by half the
    # pixel size in each direction so that the center of each pixel is properly aligned
    # with its coords
    x_half_pixel = (coords_1_1d[1] - coords_1_1d[0]) / 2
    y_half_pixel = (coords_2_1d[1] - coords_2_1d[0]) / 2
    img_extent = [
        x_low - x_half_pixel,      # left
        x_high + x_half_pixel,     # right
        y_high + y_half_pixel,     # bottom (origin='upper': img[h-1] here = y_max galvo)
        y_low - y_half_pixel,      # top (origin='upper': img[0] here = y_min galvo)

    ]

    return coords_1, coords_2, coords_1_1d, coords_2_1d, img_extent


def get_scan_cross_2d(
    center_1, center_2, scan_range_1, scan_range_2, num_steps_1, num_steps_2
):
    """Scan in a cross pattern. The first axis will be scanned while the second is held at its center,
    then the second axis will be scanned while the first is held at its center. This is useful for optimization

    Parameters
    ----------
    center_1 : numeric
        Center of the scan along the first axis
    center_2 : numeric
        Center of the scan along the second axis
    scan_range_1 : numeric
        Full range of the scan along the first axis
    scan_range_2 : numeric
        Full range of the scan along the second axis
    num_steps_1 : int
        Number of steps along the first axis
    num_steps_2 : int
        Number of steps along the second axis

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the cross scan
    array(numeric)
        Values to write to the second axis for the cross scan
    array(numeric)
        First-axis coordinates (i.e. coordinates scanned through while second axis is fixed)
    array(numeric)
        Second-axis coordinates (i.e. coordinates scanned through while first axis is fixed)
    """

    coords_1_1d = get_scan_1d(center_1, scan_range_1, num_steps_1)
    coords_2_1d = get_scan_1d(center_2, scan_range_2, num_steps_2)

    coords_1 = np.concatenate([coords_1_1d, np.full(num_steps_2, center_1)])
    coords_2 = np.concatenate([np.full(num_steps_1, center_2), coords_2_1d])

    return coords_1, coords_2, coords_1_1d, coords_2_1d


def get_scan_cross_3d(
    center_1,
    center_2,
    center_3,
    scan_range_1,
    scan_range_2,
    scan_range_3,
    num_steps_1,
    num_steps_2,
    num_steps_3,
):
    """Extension of get_scan_cross_2d to 3D"""

    coords_1_1d = get_scan_1d(center_1, scan_range_1, num_steps_1)
    coords_2_1d = get_scan_1d(center_2, scan_range_2, num_steps_2)
    coords_3_1d = get_scan_1d(center_3, scan_range_3, num_steps_3)

    coords_1 = np.concatenate(
        [coords_1_1d, np.full(num_steps_2 + num_steps_3, center_1)]
    )
    coords_2 = np.concatenate(
        [np.full(num_steps_1, center_2), coords_2_1d, np.full(num_steps_3, center_2)]
    )
    coords_3 = np.concatenate(
        [np.full(num_steps_1 + num_steps_2, center_3), coords_3_1d]
    )

    return coords_1, coords_2, coords_3, coords_1_1d, coords_2_1d, coords_3_1d


def get_scan_one_axis_2d(center_1, center_2, scan_range_1, num_steps_1):
    """Scan through the first axis, keeping the second fixed

    Parameters
    ----------
    center_1 : numeric
        Center of the scan along the first axis
    center_2 : numeric
        Center of the scan along the second axis
    scan_range_1 : numeric
        Full range of the scan along the first axis
    num_steps_1 : int
        Number of steps along the first axis

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the scan
    array(numeric)
        Values to write to the second axis for the scan
    """

    coords_1 = get_scan_1d(center_1, scan_range_1, num_steps_1)
    coords_2 = np.full(num_steps_1, center_2)
    return coords_1, coords_2


def get_scan_circle_2d(center_1, center_2, radius, num_steps):
    """Get coordinates for a scan around in a circle. Useful for checking galvo alignment

    Parameters
    ----------
    center_1 : numeric
        First-axis center of the circle
    center_2 : numeric
        First-axis center of the circle
    radius : numeric
        Radius of the circle
    num_steps : int
        Number of steps to discretize the circle into

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the scan
    array(numeric)
        Values to write to the second axis for the scan
    """

    angles = np.linspace(0, 2 * np.pi, num_steps)
    coords_1 = center_1 + (radius * np.sin(angles))
    coords_2 = center_2 + (radius * np.cos(angles))
    return coords_1, coords_2


def get_scan_two_point_2d(first_coord_1, first_coord_2, second_coord_1, second_coord_2):
    """Flip back an forth between two points - designed to be run continuously

    Parameters
    ----------
    first_coord_1 : numeric
        First point, first axis coordinate
    first_coord_2 : numeric
        First point, second axis coordinate
    second_coord_1 : numeric
        Second point, first axis coordinate
    second_coord_2 : numeric
        Second point, second axis coordinate

    Returns
    -------
    array(numeric)
        Values to write to the first axis for the scan
    array(numeric)
        Values to write to the second axis for the scan
    """

    # Sometimes a minimum number of points is required in a stream, so
    # return a list of 64 coords to be safe
    coords_1 = [first_coord_1, second_coord_1] * 32
    coords_2 = [first_coord_2, second_coord_2] * 32
    return coords_1, coords_2


# endregion


def analyze_hysteresis(target_positions, actual_positions):
    """
    Fit a hysteresis model to the data, print the fitted coefficients, and plot the results.

    Parameters
    ----------
    target_positions : ndarray
        Array of target positions (voltages).
    actual_positions : ndarray
        Array of actual measured positions (voltages).

    Returns
    -------
    tuple
        Fitted coefficients (a, b, c) of the hysteresis model.
    """

    # Define the quadratic model for hysteresis fitting
    def hysteresis_model(x, a, b, c):
        return a * x**2 + b * x + c

    # # Define the cubic model for hysteresis fitting
    # def hysteresis_model_cubic(x, a, b, c, d):
    #     return a * x**3 + b * x**2 + c * x + d

    # Flatten the 2D arrays for fitting
    target_positions_flat = target_positions.flatten()
    actual_positions_flat = actual_positions.flatten()
    print(actual_positions_flat)
    # Fit the model to the data
    popt, pcov = curve_fit(
        hysteresis_model, target_positions_flat, actual_positions_flat
    )

    # Extract the coefficients
    a, b, c = popt
    print(f"Fitted coefficients: a={a}, b={b}, c={c}")

    # Calculate residuals
    residuals = actual_positions_flat - hysteresis_model(target_positions_flat, *popt)
    residual_sum_of_squares = np.sum(residuals**2)
    total_sum_of_squares = np.sum(
        (actual_positions_flat - np.mean(actual_positions_flat)) ** 2
    )
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    print(f"R-squared: {r_squared}")

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Scatter plot of actual positions vs. target positions
    plt.subplot(1, 2, 1)
    plt.plot(target_positions_flat, actual_positions_flat, "o", label="Measured Data")
    plt.plot(
        target_positions_flat,
        hysteresis_model(target_positions_flat, *popt),
        "-",
        label="Fitted Curve",
    )
    plt.xlabel("Target Position")
    plt.ylabel("Actual Position")
    plt.legend()
    plt.title("Hysteresis Fit")

    # Plot residuals
    plt.subplot(1, 2, 2)
    plt.plot(target_positions_flat, residuals, "o")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Target Position")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")

    plt.tight_layout()
    plt.show()

    return popt


# Example usage
if __name__ == "__main__":
    green_laser = "laser_INTE_520"
    yellow_laser = "laser_OPTO_589"
    red_laser = "laser_COBO_638"
    green_laser_aod = f"{green_laser}_aod"
    red_laser_aod = f"{red_laser}_aod"

    print(get_drift(coords_key=CoordsKey.SAMPLE))

    ### Analyze hysteresis

    # # Example data (replace with your actual data)
    # target_positions = np.array(
    #     [
    #         [66.912, 94.016],
    #         [75.951, 103.95],
    #         [84.271, 113.12],
    #         [93.216, 121.95],
    #         [101.48, 131.22],
    #         [109.155, 139.9],
    #     ]
    # )
    # actual_positions = np.array(
    #     [
    #         [64.39, 92.966],
    #         [72.086, 101.0],
    #         [80.055, 110.0],
    #         [88.34, 119.3],
    #         [97.109, 128.3],
    #         [106.106, 137.0],
    #     ]
    # )

    # analyze_hysteresis(target_positions, actual_positions)
