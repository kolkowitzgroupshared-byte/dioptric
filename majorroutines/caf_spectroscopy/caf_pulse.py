# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on February 3, 2026

@author: jchen-1
"""

import time
import traceback
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb


def main():

    dm_folder = common.get_data_manager_folder()
    timestamp = dm.get_time_stamp()

    pulse_streamer = tb.get_server_pulse_streamer()

    # Procedure:
    # 0. Start recording counts on the APD for tinit seconds before turning on the laser
    #       Continue collection for tlaser + tdecay seconds
    # 1. Shine K nm laser onto the sample for duration tlaser seconds

    # Sanity check: plot pulse sequence

    # Plot data

    raw_data = {
        "num_steps": 0,
    }
    tb.reset_cfm()
    raw_data |= {
        "timestamp": timestamp,
    }
    repr_caf_name = "irr-4"
    file_path = dm.get_file_path(__file__, timestamp, repr_caf_name)
    dm.save_raw_data(raw_data, file_path)
    return
