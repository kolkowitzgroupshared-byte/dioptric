# -*- coding: utf-8 -*-
"""
Tools for managing our experimental database

Created November 15th, 2023

@author: mccambria
@author:
"""

# region Imports and constants

import copy
import io
import os
import socket
import sys
import time
import traceback
from dataclasses import fields
from datetime import datetime, timezone
from enum import Enum, auto
from io import BytesIO
from pathlib import Path

import numpy as np
import orjson  # orjson is faster and more lightweight than ujson, but can't write straight to file
import ujson  # usjson is faster than standard json library
from git import Repo

# from utils import _cloud_box as cloud
# fmt: off
# Select your cloud backend here. Box was used up until May 2025. Nas is used currently
from utils import _cloud_nas as cloud

# fmt: on
from utils import common, widefield
from utils.constants import NVSig
from utils.search_index import get_file_parent

data_manager_folder = common.get_data_manager_folder()
nvdata_dir = common.get_nvdata_dir()


# endregion
# region Save functions


def get_time_stamp():
    """Get a formatted timestamp for file names and metadata.

    Returns:
        string: <year>_<month>_<day>-<hour>_<minute>_<second>
    """

    timestamp = str(datetime.now())
    timestamp = timestamp.split(".")[0]  # Keep up to seconds
    timestamp = timestamp.replace(":", "_")  # Replace colon with dash
    timestamp = timestamp.replace("-", "_")  # Replace dash with underscore
    timestamp = timestamp.replace(" ", "-")  # Replace space with dash
    return timestamp


def get_file_path(source_file, time_stamp, name, subfolder=None):
    """Get the file path to save to. This will be in a subdirectory of nvdata

    Parameters
    ----------
    source_file : string
        __file__ variable of the caller file - this will be parsed to get the
        name of the subdirectory to write to
    time_stamp : string
        Formatted timestamp to include in the file name
    name : string
        The full file name consists of <timestamp>_<name>.txt
        Ext may be modified by specific save functions
    subfolder : string, optional
        Subfolder to save to under file name, by default None

    Returns
    -------
    Path
        Path to save to
    """

    pc_name = socket.gethostname()
    branch_name = _get_branch_name()
    source_name = Path(source_file).stem
    date_folder = "_".join(time_stamp.split("_")[0:2])  # yyyy_mm

    path_from_nv_data = f"pc_{pc_name}/branch_{branch_name}/{source_name}/{date_folder}"
    folder_path = nvdata_dir / path_from_nv_data

    if subfolder is not None:
        folder_path = folder_path / subfolder

    file_name = f"{time_stamp}-{name}.txt"

    return folder_path / file_name


def save_figure(fig, file_path):
    """Save a matplotlib figure as a png.

    Params:
        fig: matplotlib.figure.Figure
            The figure to save
        file_path: Path
            Complete file path to save to. A new Path object will be created
            with the appropriate extension
    """

    # Write to bytes then upload that to the cloud
    ext = "png"
    file_path = file_path.with_suffix(f".{ext}")
    content = BytesIO()
    # fig.savefig(content, format=ext)
    fig.savefig(content, format=ext, dpi=300, bbox_inches="tight")
    cloud.upload(file_path, content.getbuffer())


def save_raw_data(raw_data, file_path, keys_to_compress=None):
    """Save raw data in the form of a dictionary to a text file. New lines
    will be printed between entries in the dictionary.

    Params:
        raw_data: dict
            The raw data as a dictionary - will be saved via JSON
        file_path: Path
            Complete file path to save to. A new Path object will be created
            with the .txt extension
        keys_to_compress: list(string)
            Keys to values in raw_data that we want to extract and save to
            a separate compressed file. Currently supports numpy arrays
    """

    # Always compress a few specific large items
    if keys_to_compress is None:
        keys_to_compress = []
    key_options = ["img_arrays", "mean_img_arrays", "counts"]
    for key in key_options:
        if key in raw_data:
            keys_to_compress.append(key)

    # start = time.time()
    file_stem = file_path.parent
    file_path_txt = file_path.with_suffix(".txt")

    # Work with a copy of the raw data to avoid mutation
    # raw_data = copy.deepcopy(raw_data)

    # Compress numpy arrays to linked file
    try:
        if len(keys_to_compress) > 0:
            # Build the object to compress
            kwargs = {}
            for key in keys_to_compress:
                kwargs[key] = raw_data[key]
            # Upload to cloud
            content = BytesIO()
            np.savez_compressed(content, **kwargs)
            file_path_npz = file_path.with_suffix(".npz")
            cloud.upload(file_path_npz, content.getbuffer())
            # Replace the value in the raw data with the name of the compressed file
            for key in keys_to_compress:
                raw_data[key] = f"{file_stem}.npz"
    except Exception:
        print(traceback.format_exc())

    # Always include the config dict
    config = common.get_config_dict()
    config_copy = copy.deepcopy(config)
    _json_escape(config_copy)
    raw_data["config"] = config_copy

    # And the OPX config dict if there is one
    opx_config = common.get_opx_config_dict()
    if opx_config is not None:
        opx_config_copy = copy.deepcopy(opx_config)
        _json_escape(opx_config_copy)
        raw_data["opx_config"] = opx_config_copy

    # Upload raw data to the cloud
    try:
        option = (
            orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
        )
        content = orjson.dumps(raw_data, option=option)
        cloud.upload(file_path_txt, content, do_add_to_search_index=True)
    except Exception:
        print(traceback.format_exc())
        # Save to local file instead
        with open(data_manager_folder / file_path_txt.name, "wb") as f:
            f.write(content)

    # stop = time.time()
    # print(stop - start)


# region Load functions


def get_raw_data(file_stem, use_cache=True, load_npz=False, allow_pickle=False):
    """Returns a dictionary containing the json object from the specified
    raw data file

    Parameters
    ----------
    file_stem : str or list(str)
        Name of the raw data file to load, w/o extension. File names are unique
        so it is not necessary to specify a folder. If list, then data in each
        file will be concatenated into one larger object
    use_cache : bool, optional
        Whether or not to use the cache.
    load_npz : bool, optional
        Whether or not to retrieve any linked compressed numpy files (.npz files).
    allow_pickle : bool, optional
        If True, allows loading object arrays from .npz files that require pickling.
        Use with caution — only enable if you trust the source.

    Returns
    -------
    dict
        Dictionary containing the json object from the specified raw data file
    """

    if isinstance(file_stem, (list, tuple)):
        file_stems = file_stem
        data = get_raw_data(file_stems[0], use_cache, load_npz, allow_pickle)
        for file_stem in file_stems[1:]:
            new_data = get_raw_data(file_stem, use_cache, load_npz, allow_pickle)
            data["num_runs"] += new_data["num_runs"]
            data["counts"] = np.append(data["counts"], new_data["counts"], axis=2)
        return data

    retrieved_from_cache = False
    if use_cache:
        try:
            with open(data_manager_folder / "cache_manifest.txt") as f:
                cache_manifest = ujson.load(f)
        except Exception:
            cache_manifest = None

        try:
            cache_entry = cache_manifest[file_stem]
            if not load_npz or cache_entry["load_npz"]:
                with open(data_manager_folder / f"{file_stem}.txt", "rb") as f:
                    data = orjson.load(f)
                retrieved_from_cache = True
        except Exception:
            pass

    if not retrieved_from_cache:
        data_bytes = cloud.download(file_stem, ".txt")
        data = orjson.loads(data_bytes)

        if load_npz:
            found_npz = False
            for key in data:
                val = data[key]
                if isinstance(val, str) and val.endswith(".npz"):
                    found_npz = True
                    break
            if found_npz:
                npz_bytes = cloud.download(file_stem, ".npz")
                npz_data = np.load(io.BytesIO(npz_bytes), allow_pickle=allow_pickle)
                data |= npz_data

    if use_cache and not allow_pickle:
        cache_manifest_updated = False
        if not retrieved_from_cache:
            if cache_manifest is None:
                cache_manifest = {}
            cached_file_stems = list(cache_manifest.keys())
            cache_manifest[file_stem] = {"load_npz": load_npz}
            cached_file_stems.append(file_stem)
            while len(cached_file_stems) > 10:
                file_stem_to_remove = cached_file_stems.pop(0)
                del cache_manifest[file_stem_to_remove]
                file_to_remove = data_manager_folder / f"{file_stem_to_remove}.txt"
                if file_to_remove.exists():
                    os.remove(file_to_remove)
            cache_manifest_updated = True
        if cache_manifest_updated:
            with open(data_manager_folder / "cache_manifest.txt", "w+") as f:
                ujson.dump(cache_manifest, f, indent=2)

        if not retrieved_from_cache:
            file_content = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
            with open(data_manager_folder / f"{file_stem}.txt", "wb") as f:
                f.write(file_content)

    valid_fields = {field.name for field in fields(NVSig)}

    if "nv_list" in data:
        nv_list = data["nv_list"]
        nv_list = [
            NVSig(**{key: nv[key] for key in nv if key in valid_fields})
            for nv in nv_list
        ]
        data["nv_list"] = nv_list

    return data


# endregion
# region Misc public functions


def get_time_stamp_from_file_name(file_name):
    """Extract the formatted timestamp from a file name

    Returns:
        string: <year>_<month>_<day>-<hour>_<minute>_<second>
    """

    file_name_split = file_name.split("-")
    time_stamp_parts = file_name_split[0:2]
    timestamp = "-".join(time_stamp_parts)
    return timestamp


def utc_from_file_name(file_name, time_zone="PT"):
    """Convert the timestamp in a file name to a UTC time code"""
    # First 19 characters are human-readable timestamp
    date_time_str = file_name[0:19]
    date_time_str += f"-{time_zone}"
    date_time = datetime.strptime(date_time_str, r"%Y_%m_%d-%H_%M_%S-%Z")
    timestamp = date_time.timestamp()
    return timestamp


# endregion
# region Private functions


def _get_branch_name():
    """Return the name of the active branch of dioptric"""
    repo_path = common.get_repo_path()
    repo = Repo(repo_path)
    return repo.active_branch.name


def _json_escape(raw_data):
    """Recursively escape a raw data object for JSON. Slow, so use sparingly"""

    # See what kind of loop we need to do through the object
    if isinstance(raw_data, dict):
        # Just get the original keys
        keys = list(raw_data.keys())
    elif isinstance(raw_data, list):
        keys = range(len(raw_data))

    for key in keys:
        val = raw_data[key]

        # Escape the value
        if isinstance(val, Path):
            raw_data[key] = str(val)
        elif isinstance(val, type):
            raw_data[key] = str(val)

        # Recursion for dictionaries and lists
        elif isinstance(val, dict) or isinstance(val, list):
            _json_escape(val)


# endregion


if __name__ == "__main__":
    file_path = data_manager_folder / "test.txt"

    # test = {"key": np.array([[1.1, 1.2], [2.1, 2.2]], dtype=np.float16)}
    # option = orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
    # content = orjson.dumps(test, option=option)
    # file_path.write_bytes(content)

    data_bytes = file_path.read_bytes()
    data = orjson.loads(data_bytes)

    sys.exit()

    file_stem = "2025_05_02-10_43_10-rubin-nv0_2025_02_26"
    # file_stem = [
    #     "2025_05_08-04_29_55-rubin-nv0_2025_02_26",
    #     # "2025_05_07-12_41_35-rubin-nv0_2025_02_26",
    # ]
    data = get_raw_data(file_stem, use_cache=False, load_npz=False)
    # print(get_file_parent(file_stem[0]))
    debu = 0
