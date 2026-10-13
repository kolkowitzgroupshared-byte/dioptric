# -*- coding: utf-8 -*-
"""
Functions, etc to be used mainly by other utils. If you're running into
a circular reference in utils, put the problem code here.

Created September 10th, 2021

@author: mccambria
"""

import copy
import importlib
import json
import os
import platform
import socket
import sys
import time
from functools import cache
from pathlib import Path

import labrad
import numpy as np

global_cxn = None

windows_nvdata_dir = Path("E:/Shared drives/Kolkowitz Lab Group/nvdata")
linux_nvdata_dir = Path.home() / "E/nvdata"


def get_nvdata_dir():
    """Returns an OS-dependent Path to the nvdata directory (configured above)"""
    os_name = platform.system().lower()
    config = get_config_dict()
    nvdata_dir = config[f"{os_name}_nvdata_path"]
    return nvdata_dir


def get_config_module(pc_name=None, reload=False):
    if pc_name is None:
        pc_name = socket.gethostname()
    pc_name = pc_name.lower()
    try:
        module_name = f"config.{pc_name}"
        module = importlib.import_module(module_name)
    except Exception as exc:  # Fallback to the default
        module_name = "config.default"
        module = importlib.import_module(module_name)
    if reload:
        module = importlib.reload(module)

    return module


@cache
def get_config_dict(pc_name=None):
    module = get_config_module(pc_name)
    return module.config


@cache
def get_opx_config_dict(pc_name=None):
    module = get_config_module(pc_name)
    try:
        return module.opx_config
    except Exception as exc:
        return None


@cache
def get_data_manager_folder():
    data_manager_folder = get_repo_path() / "data_manager"
    if not data_manager_folder.is_dir():
        data_manager_folder.mkdir()
    return data_manager_folder


@cache
def get_labrad_logging_folder():
    pc_name = socket.gethostname()
    return get_nvdata_path() / f"pc_{pc_name}" / "labrad_logging"


@cache
def get_default_email():
    config = get_config_dict()
    return config["default_email"]


@cache
def _get_os_config_val(key):
    os_name_lower = platform.system().lower()  # windows or linux
    config = get_config_dict()
    val = config[f"{os_name_lower}_{key}"]
    return val


@cache
def get_nvdata_path():
    """Returns an OS-dependent Path to the nvdata directory"""
    return _get_os_config_val("nvdata_path")


@cache
def get_repo_path():
    """Returns an OS-dependent Path to the repo directory"""
    return _get_os_config_val("repo_path")


@cache
def get_server(server_key):
    server_name = get_server_name(server_key)
    if server_name is None:
        return None
    else:
        cxn = labrad_connect()
        return cxn[server_name]


@cache
def get_server_by_name(server_name):
    try:
        cxn = labrad_connect()
        return cxn[server_name]
    except Exception:
        return None


@cache
def get_server_name(server_key):
    config = get_config_dict()
    confg_servers = config["Servers"]
    if server_key not in confg_servers:
        return None
    server_name = confg_servers[server_key]
    return server_name


# region LabRAD registry utilities - mostly deprecated in favor of config file


def labrad_connect():
    """Return a labrad connection with default username and password"""
    global global_cxn
    if global_cxn is None:
        global_cxn = labrad.connect(username="", password="")
    return global_cxn


def set_registry_entry(directory, key, value):
    """Set an entry in the LabRAD registry"""
    cxn = labrad_connect()
    p = cxn.registry.packet()
    p.cd("", *directory)
    p.set(key, value)
    return p.send()["set"]


def get_registry_entry(directory, key):
    """Get an entry from the LabRAD registry"""
    cxn = labrad_connect()
    p = cxn.registry.packet()
    p.cd("", *directory)
    p.get(key)
    return p.send()["get"]


# endregion

if __name__ == "__main__":
    nas = get_nvdata_dir()
    print(os.listdir(nas))
