# -*- coding: utf-8 -*-
"""
Search for NV triplet-to-singlet wavelength

Created on August 9th, 2025

@author: jchen-1
"""

import sys
import time
from pathlib import Path
from random import shuffle

import numpy as np
from matplotlib import pyplot as plt

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb


def main():
    timestamp = dm.get_time_stamp()
    date, _ = timestamp.split("-")
    year, month, _ = date.split("_")

    data_path = r"G:\nvdata\pc_Nuclear\branch_master\test_data"
    month_path = f"{data_path}/{year}_{month}/{timestamp}"

    if not Path(month_path).is_dir():
        Path(month_path).mkdir(parents=True, exist_ok=True)

    raw_data = np.linspace(0, 1, 1000)
    np.save(f"{month_path}/{timestamp}_dummy.npy", raw_data)

    parameters = {
        "num_steps": 1000,
        "num_runs": 2,
    }
    parameters |= {
        "timestamp": timestamp,
    }

    repr_nv_name = "dummy_sample"
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_name)
    dm.save_raw_data(parameters, file_path)

    tisapph = tb.get_tisapph()
    print(tisapph)
    return


if __name__ == "__main__":
    main()
    # kpl.init_kplotlib()

# kpl.show(block=True)
