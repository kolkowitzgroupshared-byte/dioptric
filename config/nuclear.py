# -*- coding: utf-8 -*-
"""
Config file for the PC nuclear

Created August 5th, 2025
@author: mccambria
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.default import config
from utils.constants import (
    ChargeStateEstimationMode,
    CollectionMode,
    CoordsKey,
    CountFormat,
    ModMode,
    PosControlMode,
    VirtualLaserKey,
)

home = Path.home()


green_laser = "laser_COBO_515"  # make labrad server for COBOLT green laser
tisapph_laser = ""  # fill this in later (labrad server for Tisapph)

# region Base config
# Add on to the default config
config |= {
    ###
    "apd_indices": [0],  # APD indices for the tagger
    "count_format": CountFormat.RAW,
    # "collection_mode": CollectionMode.CAMERA,
    "collection_mode": CollectionMode.COUNTER,
    # "charge_state_estimation_mode": ChargeStateEstimationMode.MLE,
    "charge_state_estimation_mode": ChargeStateEstimationMode.THRESHOLDING,
    "disable_z_drift_compensation": False,
    ###
    # Common durations are in ns
    "CommonDurations": {
        "default_pulse_duration": 1000,
        "aod_access_time": 11e3,  # access time in specs is 10us
        "widefield_operation_buffer": 1e3,
        "uwave_buffer": 16,
        "iq_buffer": 0,
        "iq_delay": 136,  # SBC measured using NVs 4/18/2025
        "temp_reading_interval": 15 * 60,  # for PID
        # "iq_delay": 140,  # SBC measured using NVs 4/18/2025
    },
    ###
    "DeviceIDs": {
        "shutter_STAN_sr474_visa": "TCPIP::192.168.0.119::inst0::INSTR",
        "multimeter_MULT_mp730028_visa": "TCPIP0::192.168.0.174::3000::SOCKET",
        "multimeter_KEIT_daq6510_visa": "TCPIP0::192.168.0.175::inst0::INSTR",
        "tisapph_M2_solstis_ip": "192.168.0.195",
        "tisapph_pump_COHE_verdi_com": "COM6",
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.178::inst0::INSTR",
        "filter_slider_THOR_ell9k_com": "",
        "pulse_gen_SWAB_82_1_ip_1": "192.168.0.111",
        "pulse_gen_SWAB_82_2_ip_2": "192.168.0.160",
        "tagger_SWAB_20_1_serial": "1740000JEH",
    },
    ###
    "Microwaves": {
        "PhysicalSigGens": {
            "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394": {"delay": 104, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_2": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_TEKT_tsg4104a": {"delay": 57},
        },
        "iq_comp_amp": 0.5,
        "iq_delay": 140,  # SBC measured using NVs 4/18/2025
        "VirtualSigGens": {
            0: {
                "physical_name": "sig_gen_STAN_sg394",
                "uwave_power": 8.7,
                "frequency": 2.730700,
                "rabi_period": 128,
            },
            # sig gen 1 is iq molulated
            1: {
                "physical_name": "sig_gen_STAN_sg394_2",
                "uwave_power": 8.7,
                "frequency": 2.730700,  # lower esr peak for both orientation
                "rabi_period": 208,
                "pi_pulse": 104,
                "pi_on_2_pulse": 56,
            },
        },
    },
    ###
    "Optics": {
        "PhysicalLasers": {
            green_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": CoordsKey.PIXEL,
            },
            tisapph_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": CoordsKey.PIXEL,
            },
        },
        "VirtualLasers": {
            # LaserKey.IMAGING: {"physical_name": green_laser, "duration": 50e6},
            VirtualLaserKey.IMAGING: {
                # "physical_name": green_laser,
                "physical_name": green_laser,  # this is the laser that appears on the imaging APD scan
                "duration": 100e6,  # this duration appears on the imaging APD scan, this value is overwritten?
            },
        },
        #
        "PulseSettings": {
            "scc_shelving_pulse": False,  # Example setting
        },  # Whether or not to include a shelving pulse in SCC
    },
    ###
    "Servers": {  # Bucket for miscellaneous servers not otherwise listed above
        "camera": "camera_NUVU_hnu512gamma",
        "thorslm": "slm_THOR_exulus_hd2",
        "slider_1": "filter_slider_THOR_ell9k_4",
        "slider_2": "filter_slider_THOR_ell9k_5",
        "slider_3": "filter_slider_THOR_ell9k_6",
        "power_meter": "power_meter_THOR_pm100d",
        "rotation_mount": "rotation_mount_THOR_ell14",
        "pulse_streamer": "pulse_gen_SWAB_82_1",
        "counter": "tagger_SWAB_20_1",
    },
    ###
    "Wiring": {
        # https://docs-be.ni.com/bundle/ni-67xx-scb-68a-labels/raw/resource/enus/371806a.pdf
        "PulseGen": {
            "do_apd_gate": 4,
            "do_laser_COBO_515_dm": 0,
            "do_laser_COBO_532_dm": 1,
            "do_laser_COBO_638_dm": 2,
        },
        "Tagger": {"di_apd_0": 1, "di_apd_gate": 2},
    },
}

# endregion
