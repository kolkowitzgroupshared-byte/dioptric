# -*- coding: utf-8 -*-
"""
Config file for the PC nuclear

Created August 5th, 2025
@author: mccambria
"""

from pathlib import Path

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

# region Base config
# Add on to the default config
config |= {
    ###
    "apd_indices": [0],  # APD indices for the tagger
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.CAMERA,
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
                # "uwave_power": 2.3,
                "uwave_power": 8.7,
                # "frequency": 2.779138,  # rubin shallow NVs O1 ms=-1
                # "frequency": 2.909381,  # rubin shallow NV O3 ms=+1
                "frequency": 2.730700,
                "rabi_period": 128,
            },
            # sig gen 1 is iq molulated
            1: {
                "physical_name": "sig_gen_STAN_sg394_2",
                "uwave_power": 8.7,
                # "uwave_power": 9.6,
                # "frequency": 2.779138,   # rubin shallow NVs O1 ms=-1
                # "frequency": 2.964545,  # rubin shallow NV O1 ms=+1
                # "frequency": 2.842478,  # rubin shallow NV O3 ms=-1
                "frequency": 2.730700,  # lower esr peak for both orientation
                "rabi_period": 208,
                "pi_pulse": 104,
                "pi_on_2_pulse": 56,
                # "rabi_period": 52,
            },
        },
    },
    ###
    ###
    "Optics": {
        "PhysicalLasers": {},
        "VirtualLasers": {},
    },
    ###
    "Servers": {  # Bucket for miscellaneous servers not otherwise listed above
        "pulse_gen": "QM_opx",
        "camera": "camera_NUVU_hnu512gamma",
        "thorslm": "slm_THOR_exulus_hd2",
        "slider_1": "filter_slider_THOR_ell9k_4",
        "slider_2": "filter_slider_THOR_ell9k_5",
        "slider_3": "filter_slider_THOR_ell9k_6",
        "power_meter": "power_meter_THOR_pm100d",
        "rotation_mount": "rotation_mount_THOR_ell14",
    },
    ###
    "Wiring": {
        # https://docs-be.ni.com/bundle/ni-67xx-scb-68a-labels/raw/resource/enus/371806a.pdf
        "Daq": {
            "ao_galvo_x": "dev1/AO31",
            "ao_galvo_y": "dev1/AO22",
            "ao_piezo_stage_P616_3c_x": "dev1/AO25",
            "ao_piezo_stage_P616_3c_y": "dev1/AO27",
            "ao_piezo_stage_P616_3c_z": "dev1/AO29",
            "ao_objective_piezo": "dev1/AO21",
            "voltage_range_factor": 10.0,
            "di_clock": "PFI12",
        },
        "Piezo_Controller_E727": {
            "piezo_controller_channel_x": 4,
            "piezo_controller_channel_y": 5,
            "piezo_controller_channel_z": 6,
            "voltage_range_factor": 10.0,
            "scaling_offset": 50.0,
            "scaling_gain": 0.5,
        },
        "PulseGen": {
            "do_apd_gate": 5,
            "do_laser_INTE_520_dm": 3,
            "do_laser_OPTO_589_dm": 3,
            "do_laser_COBO_638_dm": 7,
            "do_sample_clock": 0,
            "do_sig_gen_BERK_bnc835_gate": 1,
            "do_sig_gen_STAN_sg394_gate": 10,
            "do_camera_trigger": 5,
        },
        "Tagger": {"di_apd_0": 2, "di_apd_1": 4, "di_apd_gate": 3, "di_clock": 1},
    },
}

# endregion
