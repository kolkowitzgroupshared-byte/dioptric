# -*- coding: utf-8 -*-
"""
Config file for the PC purcell

Created July 20th, 2023
@author: mccambria
@author: sbchand
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

# region Widefield calibration coords

green_laser = "laser_INTE_520"
yellow_laser = "laser_OPTO_589"
red_laser = "laser_COBO_638"
green_laser_aod = "laser_INTE_520_aod"
red_laser_aod = "laser_COBO_638_aod"


calibration_coords_pixel = [[14.043, 37.334],[106.538, 237.374],[218.314, 23.302]]
calibration_coords_green = [[119.248, 119.584],[111.265, 95.774],[95.933, 118.969]]
calibration_coords_red = [[82.15, 82.282],[76.463, 62.52],[63.114, 80.587]]

# Create the dictionaries using the provided lists
calibration_coords_nv1 = {
    CoordsKey.PIXEL: calibration_coords_pixel[0],
    green_laser_aod: calibration_coords_green[0],
    red_laser_aod: calibration_coords_red[0],
}

calibration_coords_nv2 = {
    CoordsKey.PIXEL: calibration_coords_pixel[1],
    green_laser_aod: calibration_coords_green[1],
    red_laser_aod: calibration_coords_red[1],
}

calibration_coords_nv3 = {
    CoordsKey.PIXEL: calibration_coords_pixel[2],
    green_laser_aod: calibration_coords_green[2],
    red_laser_aod: calibration_coords_red[2],
}

pixel_to_sample_affine_transformation_matrix = [
    [0.01476835, -0.00148369, -1.42104908],
    [0.00140560, 0.01479702, -1.73286644],
]
# endregion
# region Base config
# Add on to the default config
config |= {
    ###
    "apd_indices": [0],  # APD indices for the tagger
    "count_format": CountFormat.RAW,
    "collection_mode": CollectionMode.CAMERA,
    "collection_mode_counter": CollectionMode.COUNTER,  # TODO: remove this line when set up in new computer
    # "charge_state_estimation_mode": ChargeStateEstimationMode.MLE,
    "charge_state_estimation_mode": ChargeStateEstimationMode.THRESHOLDING,
    "windows_repo_path": home / "GitHub/dioptric",
    "disable_z_drift_compensation": False,
    ###
    # Common durations are in ns
    "CommonDurations": {
        "default_pulse_duration": 1000,
        "aod_access_time":3e3,  # access time in specs is 10us
        "widefield_operation_buffer": 1e3,
        "uwave_buffer": 0,
        "iq_buffer": 0,
        # "iq_delay": 136,  # SBC measured using NVs 4/18/2025
        "iq_delay": 140,  # 
    },
    ###
    "DeviceIDs": {
        "arb_wave_gen_visa_address": "TCPIP0::128.104.ramp_to_zero_duration.119::5025::SOCKET",
        "daq0_name": "Dev1",
        "filter_slider_THOR_ell9k_com": "COM13",
        "gcs_dll_path": home
        / "GitHub/dioptric/servers/outputs/GCSTranslator/PI_GCS2_DLL_x64.dll",
        "objective_piezo_model": "E709",
        "objective_piezo_serial": "0119008970",
        "piezo_controller_E727_model": "E727",
        "piezo_controller_E727_serial": "0121089079",
        "pulse_gen_SWAB_82_ip_1": "192.168.0.111",
        "pulse_gen_SWAB_82_ip_2": "192.168.0.160",
        "rotation_stage_THOR_ell18k_com": "COM8",
        "sig_gen_BERK_bnc835_visa": "TCPIP::128.104.ramp_to_zero_duration.114::inst0::INSTR",
        "sig_gen_STAN_sg394_0_visa": "TCPIP::192.168.0.120::inst0::INSTR",
        "sig_gen_STAN_sg394_1_visa": "TCPIP::192.168.0.121::inst0::INSTR",
        "sig_gen_STAN_sg394_2_visa": "TCPIP::192.168.0.178::inst0::INSTR",
        "sig_gen_STAN_sg394_3_visa": "TCPIP::192.168.0.177::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.ramp_to_zero_duration.112::5025::SOCKET",
        "tagger_SWAB_20_1_serial": "1740000JEH",
        "tagger_SWAB_20_2_serial": "1948000SIP",
        "QM_opx_args": {
            "host": "192.168.0.117",
            "port": 9510,
            "cluster_name": "kolkowitz_nv_lab",
        },
        "power_supply_RNS_ngc103_visa": "TCPIP::192.168.000.130::INSTR",
    },
    ###
    "Microwaves": {
        "PhysicalSigGens": {
            "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_0": {"delay": 104, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_1": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_3": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_TEKT_tsg4104a": {"delay": 57},
        },
        "iq_comp_amp": 0.5,
        "iq_delay": 140,
        "VirtualSigGens": {
            0: {
                "physical_name": "sig_gen_STAN_sg394_0",
                "uwave_power": 11.0,
                "frequency": 2.771367, #49G
                # "frequency": 2.747151,
                # "frequency": 2.709799,  #62G
                # "frequency":2.963189,
                # "frequency": 2.917151,
                # "frequency": 2.8082,
                "rabi_period": 256,
                "pi_pulse": 128,
                # "pi_pulse": 240,
                "pi_on_2_pulse": 64,
                # "frequency": 2.935030,
                # "rabi_period": 112,
                # "pi_pulse": 56,
                # "pi_on_2_pulse": 28,
            },
            1: {
                "physical_name": "sig_gen_STAN_sg394_1",
                "uwave_power": 11.0,
                # "frequency": 2.917151,
                "frequency": 2.810689,#49G
                # "frequency": 2.982049,
                # "frequency": 2.828210, 
                # "frequency": 2.816912, #62G
                # "frequency": 2.8252,
                "rabi_period": 192,
                "pi_pulse": 88,
                "pi_on_2_pulse": 44,
            },
            2: {
                "physical_name": "sig_gen_STAN_sg394_3",
                "uwave_power": 11.0,
                "frequency": 0.158,
                "rabi_period": 176,
                "pi_pulse": 2000,
                "pi_on_2_pulse": 100,
            },
        },
    },
    ###
    "Camera": {
        "server_name": "camera_NUVU_hnu512gamma",
        "resolution": (512, 512),
        "spot_radius": 2.5,  # Radius for integrating NV counts in a camera image
        "bias_clamp": 300,  # (changing this won't actually change the value on the camera currently)
        "em_gain": 5000,
        # "em_gain": 10,
        "temp": -60,
        "timeout": 60e3,  # ms
        # "timeout": -1,  # No timeout
        # Readout mode specifies EM vs conventional, as well as vertical and horizontal readout frequencies.
        # See camera server file for details
        "readout_mode": 1,  # 16 for double horizontal readout rate (em mode)
        # "readout_mode": 6,  # Fast conventional
        "roi": (122, 126, 250, 250),  # offsetX, offsetY, width, height
        # "roi": None,  # offsetX, offsetY, width, height
        "scale": 5 / 0.6,  # pixels / micron
    },
    ###
    "Optics": {
        "PhysicalLasers": {
            green_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": green_laser_aod,
            },
            red_laser: {
                "delay": 0,
                "mod_mode": ModMode.DIGITAL,
                "positioner": red_laser_aod,
            },
            yellow_laser: {
                "delay": 0,
                "mod_mode": ModMode.ANALOG,
            },
        },
        "VirtualLasers": {
            # LaserKey.IMAGING: {"physical_name": green_laser, "duration": 50e6},
            VirtualLaserKey.IMAGING: {
                "physical_name": green_laser,
                # "duration": 12e6,
                "duration": 12e6,
            },
            # SBC: created for calibration only
            VirtualLaserKey.RED_IMAGING: {
                "physical_name": red_laser,
                "duration": 1e6,
            },
            VirtualLaserKey.SPIN_READOUT: {
                "physical_name": green_laser,
                "duration": 200,
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 10e3},
            VirtualLaserKey.CHARGE_POL: {
                "physical_name": green_laser,
                "duration": 1e3,  # Works better for Deep NVs (Johnson)
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 60},
            VirtualLaserKey.SPIN_POL: {
                "physical_name": green_laser,
                "duration": 10e3,
            },
            VirtualLaserKey.SHELVING: {
                "physical_name": green_laser,
                "duration": 60,
            },
            VirtualLaserKey.ION: {
                "physical_name": red_laser,
                "duration": 1e3,
            },
            # SCC: 180 mW, 0.13 V, no shelving
            # LaserKey.SCC: {"physical_name": red_laser, "duration": 248},
            VirtualLaserKey.SCC: {
                "physical_name": red_laser,
                "duration": 124,
            },
            # LaserKey.SCC: {"physical_name": green_laser, "duration": 200},
            VirtualLaserKey.WIDEFIELD_SHELVING: {
                "physical_name": yellow_laser,
                "duration": 60,
            },
            VirtualLaserKey.WIDEFIELD_IMAGING: {
                "physical_name": yellow_laser,
                "duration": 12e6,
                # "duration": 24e6,
            },
            # LaserKey.WIDEFIELD_SPIN_POL: {"physical_name": yellow_laser, "duration": 10e3},
            VirtualLaserKey.WIDEFIELD_SPIN_POL: {
                "physical_name": yellow_laser,
                "duration": 100e3,
            },
            # LaserKey.WIDEFIELD_SPIN_POL: {"physical_name": yellow_laser, "duration": 1e6},
            VirtualLaserKey.WIDEFIELD_CHARGE_READOUT: {
                "physical_name": yellow_laser,
                # "duration": 200e6,
                # "duration": 60e6,
                "duration": 50e6,
                # "duration": 24e6,  # for red calibration
            },
            # LaserKey.WIDEFIELD_CHARGE_READOUT: {"physical_name": yellow_laser, "duration": 100e6},
        },
        #
        "PulseSettings": {
            "scc_shelving_pulse": False,  # Example setting
        },  # Whether or not to include a shelving pulse in SCC
    },
    ###
    "Positioning": {
        "Positioners": {
            CoordsKey.SAMPLE: {
                "physical_name": "pos_xyz_PI_p616_3c",
                "control_mode": PosControlMode.STREAM,
                "delay": int(1e6),  # 5 ms for PIFOC xyz
                "nm_per_unit": 1000,
                "optimize_range": 0.09,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
            CoordsKey.Z: {
                "physical_name": "pos_xyz_PI_p616_3c",
                "control_mode": PosControlMode.STREAM,
                "delay": int(1e6),  # 5 ms for PIFOC xyz
                "nm_per_unit": 1000,
                # "optimize_range": 0.09,
                "optimize_range": 0.24,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
            green_laser_aod: {
                "control_mode": PosControlMode.SEQUENCE,
                "delay": int(400e3),  # 400 us for galvo
                "nm_per_unit": 1000,
                "optimize_range": 1.2,
                "units": "MHz",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
                "aod": True,
            },
            red_laser_aod: {
                "control_mode": PosControlMode.SEQUENCE,
                "delay": int(400e3),  # 400 us for galvo
                "nm_per_unit": 1000,
                "optimize_range": 2.4,
                "units": "MHz",
                "opti_virtual_laser_key": VirtualLaserKey.ION,
                "aod": True,
            },
        },
        "calibration_coords_nv1": calibration_coords_nv1,
        "calibration_coords_nv2": calibration_coords_nv2,
        "calibration_coords_nv3": calibration_coords_nv3,
        "pixel_to_sample_affine_transformation_matrix": pixel_to_sample_affine_transformation_matrix,
    },
    ###
    "Servers": {  # Bucket for miscellaneous servers not otherwise listed above
        "pulse_gen": "QM_opx",
        "camera": "camera_NUVU_hnu512gamma",
        "thorslm": "slm_THOR_exulus_hd2",
        "pulse_streamer": "pulse_gen_SWAB_82",
        "counter": "tagger_SWAB_20",
    },
    ###
    "Wiring": {
        "Daq": {
            # https://docs-be.ni.com/bundle/ni-67xx-scb-68a-labels/raw/resource/enus/371806a.pdf
            "ao_galvo_x": "dev1/AO22",
            "ao_galvo_y": "dev1/AO31",
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
            # "do_laser_INTE_520_dm": 3,
            # "do_laser_OPTO_589_dm": 3,
            # "do_laser_COBO_638_dm": 7,
            # "do_sig_gen_BERK_bnc835_gate": 1,
            # "do_sig_gen_STAN_sg394_gate": 10,
            # "do_apd_gate": 5,
            # "do_sample_clock": 0,
            # "do_camera_trigger": 5,
            # clocks / gates
            "do_sample_clock": 0,  # 125 MHz-compatible sample clock out to Tagger
            "do_apd_gate": 1,  # gate line to Tagger
            # "do_camera_trigger": 6,  # optional
            "do_laser_INTE_520_dm": 2,  # green  TTL
            "do_laser_COBO_638_dm": 3,  # red TTL
            # microwaves (TTL gate to SGs)
            # "do_sig_gen_BERK_bnc835_gate": 4,
            "do_sig_gen_STAN_sg394_1_gate": 4,
            "do_sig_gen_STAN_sg394_0_gate": 5,
            # analog (for the yellow AOM amplitude)
            "ao_laser_OPTO_589_am": 0,  # yellow analog modulation
        },
        "Tagger": {
            "di_clock": 1,
            "di_apd_gate": 2,
            "di_apd_0": 3,
            "di_apd_1": 4,
        },
    },
}

# endregion


# region OPX config

default_pulse_duration = config["CommonDurations"]["default_pulse_duration"]
default_int_freq = 75e6
virtual_sig_gens_dict = config["Microwaves"]["VirtualSigGens"]
num_sig_gens = len(virtual_sig_gens_dict)
rabi_period_0 = virtual_sig_gens_dict[0]["rabi_period"]
pi_pulse_0 = virtual_sig_gens_dict[0]["pi_pulse"]
pi_on_2_pulse_0 = virtual_sig_gens_dict[0]["pi_on_2_pulse"]
rabi_period_1 = virtual_sig_gens_dict[1]["rabi_period"]
pi_pulse_1 = virtual_sig_gens_dict[1]["pi_pulse"]
pi_on_2_pulse_1 = virtual_sig_gens_dict[1]["pi_on_2_pulse"]
rabi_period_2 = virtual_sig_gens_dict[2]["rabi_period"]
pi_pulse_2 = virtual_sig_gens_dict[2]["pi_pulse"]
pi_on_2_pulse_2 = virtual_sig_gens_dict[2]["pi_on_2_pulse"]
ramp_to_zero_duration = 64
virtual_lasers_dict = config["Optics"]["VirtualLasers"]
iq_buffer = config["CommonDurations"]["iq_buffer"]
iq_delay = config["CommonDurations"]["iq_delay"]


opx_config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": 0},
                2: {"offset": 0.0, "delay": 0},
                3: {"offset": 0.0, "delay": 0},
                4: {"offset": 0.0, "delay": 0},
                5: {"offset": 0.0, "delay": 0},
                6: {"offset": 0.0, "delay": 0},
                7: {"offset": 0.0, "delay": 0},
                8: {"offset": 0.0, "delay": 0},
                9: {"offset": 0.0, "delay": 0},
                10: {"offset": 0.0, "delay": 0},
            },
            "digital_outputs": {
                1: {},  #
                2: {},  #
                3: {},  #
                4: {},  #
                5: {},  #
                6: {},  #
                7: {},  #
                8: {},  #
                9: {},  #
                10: {},  #
            },
            "analog_inputs": {
                1: {"offset": 0},  # APD0
                2: {"offset": 0},  # APD1
            },
        },
    },
    "elements": {
        # region Bare channels
        "do1": {
            "digitalInputs": {"chan": {"port": ("con1", 1), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do2": {
            "digitalInputs": {"chan": {"port": ("con1", 2), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do3": {
            "digitalInputs": {"chan": {"port": ("con1", 3), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do4": {
            "digitalInputs": {"chan": {"port": ("con1", 4), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do5": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do6": {
            "digitalInputs": {"chan": {"port": ("con1", 6), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do7": {
            "digitalInputs": {"chan": {"port": ("con1", 7), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do8": {
            "digitalInputs": {"chan": {"port": ("con1", 8), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do9": {
            "digitalInputs": {"chan": {"port": ("con1", 9), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "do10": {
            "digitalInputs": {"chan": {"port": ("con1", 10), "delay": 0, "buffer": 0}},
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "ao1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao3": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao4": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao5": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao6": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao7": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao8": {
            "singleInput": {"port": ("con1", 8)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao9": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        "ao10": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": default_int_freq,
            "operations": {"cw": "ao_cw", "off": "ao_off"},
        },
        # endregion
        # region Actual "elements", or physical things to control
        "do_laser_COBO_638_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 1), "delay": 16, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "scc": "do_scc",
                "ion": "do_ion",
            },
        },
        "ao_laser_OPTO_589_am": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "operations": {
                "on": "yellow_imaging",
                "off": "ao_off",
                "charge_readout": "yellow_charge_readout",
                "spin_pol": "yellow_spin_pol",
                "shelving": "yellow_shelving",
            },
        },
        "ao_laser_OPTO_589_am_sticky": {
            "singleInput": {"port": ("con1", 7)},
            "intermediate_frequency": 0,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "on": "yellow_imaging",
                "off": "ao_off",
                "charge_readout": "yellow_charge_readout",
            },
        },
        "do_laser_INTE_520_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 4), "delay": 0, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "charge_pol": "do_charge_pol",
                "spin_pol": "do_green_spin_pol",
                "shelving": "do_shelving",
                "scc": "do_scc",
            },
        },
        "do_sig_gen_STAN_sg394_0_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 9), "delay": iq_delay, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_0",
                "pi_on_2_pulse": "do_pi_on_2_pulse_0",
            },
        },
        "do_sig_gen_STAN_sg394_1_dm": {
            "digitalInputs": {
                # "chan": {"port": ("con1", 10), "delay": 0, "buffer": 0}
                "chan": {"port": ("con1", 10), "delay": iq_delay, "buffer": 0}
            },
            "operations": {
                "iq_test": "do_iq_test",
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_1",
                "pi_on_2_pulse": "do_pi_on_2_pulse_1",
            },
        },
        "do_sig_gen_STAN_sg394_3_dm": {
            "digitalInputs": {"chan": {"port": ("con1", 3), "delay": iq_delay, "buffer": 0}},
            "operations": {
                "on": "do_on",
                "off": "do_off",
                "pi_pulse": "do_pi_pulse_2",
                "pi_on_2_pulse": "do_pi_on_2_pulse_2",
            },
        },
        # region Microwave iq modulation
        # Additional operations are generated algorithmically with generate_iq_pulses()
        "ao_sig_gen_STAN_sg394_0_i": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_0",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_0",
            },
        },
        "ao_sig_gen_STAN_sg394_0_q": {
            "singleInput": {"port": ("con1", 8)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_0",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_0",
            },
        },
        "ao_sig_gen_STAN_sg394_1_i": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_1",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_1",
            },
        },
        "ao_sig_gen_STAN_sg394_1_q": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_1",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_1",
            },
        },
        ### SBC: added extra I/Q pair for sg394_2
        ### NOTE: these elements currently map to the *same physical AO ports*
        ### (con1,9) and (con1,10) already used by sg394_1.
        ### That means sg394_1 and sg394_2 cannot be active simultaneously.
        ### If you intend to use them independently, assign new AO ports.
        "ao_sig_gen_STAN_sg394_3_i": {
            "singleInput": {"port": ("con1", 9)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_2",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_2",
            },
        },
        "ao_sig_gen_STAN_sg394_3_q": {
            "singleInput": {"port": ("con1", 10)},
            "intermediate_frequency": 0,
            "operations": {
                "iq_test": "iq_test",
                "on": "ao_cw",
                "off": "ao_off",
                "pi_pulse": "ao_iq_pi_pulse_2",
                "pi_on_2_pulse": "ao_iq_pi_on_2_pulse_2",
            },
        },
        # endregion
        "do_camera_trigger": {
            "digitalInputs": {"chan": {"port": ("con1", 5), "delay": 0, "buffer": 0}},
            "sticky": {
                "analog": True,
                "digital": True,
                "duration": ramp_to_zero_duration,
            },
            "operations": {"on": "do_on", "off": "do_off"},
        },
        "ao_laser_COBO_638_x": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 75e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw": "red_aod_cw-scc",
                "aod_cw-opti": "red_aod_cw-opti",
                "aod_cw-ion": "red_aod_cw-ion",
                "aod_cw-scc": "red_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        "ao_laser_COBO_638_y": {
            "singleInput": {"port": ("con1", 6)},
            "intermediate_frequency": 75e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw": "red_aod_cw-scc",
                "aod_cw-opti": "red_aod_cw-opti",
                "aod_cw-ion": "red_aod_cw-ion",
                "aod_cw-scc": "red_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        "ao_laser_INTE_520_x": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": 110e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw": "green_aod_cw-charge_pol",
                "aod_cw-opti": "green_aod_cw-opti",
                "aod_cw-charge_pol": "green_aod_cw-charge_pol",
                "aod_cw-spin_pol": "green_aod_cw-spin_pol",
                "aod_cw-shelving": "green_aod_cw-shelving",
                "aod_cw-scc": "green_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        "ao_laser_INTE_520_y": {
            "singleInput": {"port": ("con1", 4)},
            "intermediate_frequency": 110e6,
            "sticky": {"analog": True, "duration": ramp_to_zero_duration},
            "operations": {
                "aod_cw": "green_aod_cw-charge_pol",
                "aod_cw-opti": "green_aod_cw-opti",
                "aod_cw-charge_pol": "green_aod_cw-charge_pol",
                "aod_cw-spin_pol": "green_aod_cw-spin_pol",
                "aod_cw-shelving": "green_aod_cw-shelving",
                "aod_cw-scc": "green_aod_cw-scc",
                "continue": "ao_off",
            },
        },
        # endregion
    },
    # region Pulses
    "pulses": {
        ### Analog
        # Green
        "green_aod_cw-opti": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-opti"},
        },
        "green_aod_cw-charge_pol": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-charge_pol"},
        },
        "green_aod_cw-spin_pol": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-spin_pol"},
        },
        "green_aod_cw-shelving": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-shelving"},
        },
        "green_aod_cw-scc": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "green_aod_cw-scc"},
        },
        # Red
        "red_aod_cw-opti": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw-opti"},
        },
        "red_aod_cw-ion": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw-ion"},
        },
        "red_aod_cw-scc": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "red_aod_cw-scc"},
        },
        # Yellow
        "yellow_imaging": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "yellow_imaging"},
        },
        "yellow_charge_readout": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "yellow_charge_readout"},
        },
        "yellow_spin_pol": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.WIDEFIELD_SPIN_POL][
                "duration"
            ],
            "waveforms": {"single": "yellow_spin_pol"},
        },
        "yellow_shelving": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.WIDEFIELD_SHELVING][
                "duration"
            ],
            "waveforms": {"single": "yellow_shelving"},
        },
        #
        "ao_cw": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "cw"},
        },
        "ao_off": {
            "operation": "control",
            "length": default_pulse_duration,
            "waveforms": {"single": "off"},
        },
        "iq_test": {
            "operation": "control",
            "length": 10000,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_pulse_0": {
            "operation": "control",
            "length": int(pi_pulse_0) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_on_2_pulse_0": {
            "operation": "control",
            "length": int(pi_on_2_pulse_0) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_pulse_1": {
            "operation": "control",
            "length": int(pi_pulse_1) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_on_2_pulse_1": {
            "operation": "control",
            "length": int(pi_on_2_pulse_1) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_pulse_2": {
            "operation": "control",
            "length": int(pi_pulse_2) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        "ao_iq_pi_on_2_pulse_2": {
            "operation": "control",
            "length": int(pi_on_2_pulse_2) + 2 * iq_buffer,
            "waveforms": {"single": "cw"},
        },
        ### Digital
        "do_iq_test": {
            "operation": "control",
            "length": 10000,
            "digital_marker": "on",
        },
        "do_on": {
            "operation": "control",
            "length": default_pulse_duration,
            "digital_marker": "on",
        },
        "do_off": {
            "operation": "control",
            "length": default_pulse_duration,
            "digital_marker": "off",
        },
        "do_short_pulse": {
            "operation": "control",
            "length": default_pulse_duration,
            "digital_marker": "square",
        },
        "do_scc": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.SCC]["duration"],
            "digital_marker": "on",
        },
        "do_ion": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.ION]["duration"],
            "digital_marker": "on",
        },
        "do_charge_pol": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.CHARGE_POL]["duration"],
            "digital_marker": "on",
        },
        "do_shelving": {
            "operation": "control",
            "length": virtual_lasers_dict[VirtualLaserKey.SHELVING]["duration"],
            "digital_marker": "on",
        },
        "do_green_spin_pol": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "on",
        },
        "do_esr_pulse": {
            "operation": "control",
            "length": 64,
            "digital_marker": "on",
        },
        "do_pi_pulse_0": {
            "operation": "control",
            "length": int(pi_pulse_0),
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_0": {
            "operation": "control",
            "length": int(pi_on_2_pulse_0),
            "digital_marker": "on",
        },
        "do_pi_pulse_1": {
            "operation": "control",
            "length": int(pi_pulse_1),
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_1": {
            "operation": "control",
            "length": int(pi_on_2_pulse_1),
            "digital_marker": "on",
        },
        "do_pi_pulse_2": {
            "operation": "control",
            "length": int(pi_pulse_2),
            "digital_marker": "on",
        },
        "do_pi_on_2_pulse_2": {
            "operation": "control",
            "length": int(pi_on_2_pulse_2),
            "digital_marker": "on",
        },
        ### Mixed
    },
    # endregion
    # region Waveforms
    ### Analog
    "waveforms": {
        # Green AOD
        "green_aod_cw-opti": {"type": "constant", "sample": 0.11},
        "green_aod_cw-charge_pol": {"type": "constant", "sample": 0.11},
        "green_aod_cw-spin_pol": {"type": "constant", "sample": 0.05},
        "green_aod_cw-shelving": {"type": "constant", "sample": 0.05},
        "green_aod_cw-scc": {"type": "constant", "sample": 0.15},
        # Red AOD
        "red_aod_cw-opti": {"type": "constant", "sample": 0.13},
        "red_aod_cw-ion": {"type": "constant", "sample": 0.13},
        "red_aod_cw-scc": {"type": "constant", "sample": 0.13},
        # Yellow AOM
        "yellow_imaging": {"type": "constant", "sample": 0.45},
        # "yellow_charge_readout": {"type": "constant", "sample": 0.2675},
        # "yellow_charge_readout": {"type": "constant", "sample": 0.2367}, #136NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.2267}, #118NVs
        # "yellow_charge_readout": {"type": "constant", "sample": 0.34947}, ## 312NV johnson
        # "yellow_charge_readout": {"type": "constant", "sample": 0.313148}, ## 230NV johnson
        # "yellow_charge_readout": {"type": "constant", "sample": 0.3084}, ## 223NV johnson
        "yellow_charge_readout": {"type": "constant", "sample": 0.299064}, ## 204NV johnson
        "yellow_spin_pol": {"type": "constant", "sample": 0.31510},
        "yellow_shelving": {"type": "constant", "sample": 0.20},
        # Other
        "aod_cw": {"type": "constant", "sample": 0.35},
        "cw": {"type": "constant", "sample": 0.5},
        "off": {"type": "constant", "sample": 0.0},
    },
    ### Digital, format is list of tuples: (on/off, ns)
    "digital_waveforms": {
        "on": {"samples": [(1, 0)]},
        "off": {"samples": [(0, 0)]},
        "square": {"samples": [(1, 100), (0, 100)]},
    },
    # endregion
}
# endregion


# def correct_pulse_params_by_phase(phase_deg, base_amp=0.5):
#     # Centralized pulse error values from bootstrap
#     pulse_errors = {
#         "phi_prime": -0.031938,
#         "chi_prime": -0.037178,
#         "phi": 0.069222,
#         "chi": 0.07584,
#         "vz": -0.016646,
#         "ez": 0.111846,
#         "epsilon_z_prime": -0.011931,
#         "nu_x_prime": -0.059049,
#         "nu_z_prime": 0.007111,
#         "epsilon_y": 0.048215,
#         "nu_x": 0.017096,
#     }

#     # Y-aligned rotation
#     tilt = pulse_errors.get("epsilon_y", 0)
#     phase_corr = -np.degrees(tilt, 0.0)
#     return phase_corr


def correct_pulse_params_by_phase(phase_deg):
    # Centralized pulse error values from bootstrap
    pulse_errors = {
        "phi_prime": -0.013781,
        "chi_prime": -0.030103,
        "phi": 0.028953,
        "chi": 0.050192,
        "vz": -0.028262,
        "ez": 0.076654,
        "epsilon_z_prime": -0.011943,
        "nu_x_prime": -0.040587,
        "nu_z_prime": -0.01359,
        "epsilon_y": 0.043858,
        "nu_x": 0.030547,
    }

    if phase_deg in [90, 270]:
        tilt = pulse_errors.get("ez", 0.0)
        phase_corr_deg = -np.degrees(tilt)
    elif phase_deg in [0, 180]:
        tilt = pulse_errors.get("vz", 0.0)
        phase_corr_deg = -np.degrees(tilt)
    else:
        phase_corr_deg = 0.0

    return phase_corr_deg


def generate_iq_pulses(pulse_names, phases):
    """Adds iq pulses to opx_config for the passed phases to match the microwave pulses
    already defined manually in the config. The pulses names that are intended for use
    are of the form f"{pulse_name}_{phase}" with duration equal to that of the pulse
    with pulse_name defined on the corresponding digital modulation element for a
    given microwave channel

    Parameters
    ----------
    pulse_names : list(str)
        List of microwave pulse names
    phases : list(int)
        List of phases in degrees. Expects integers.
    """
    # Define the waveforms
    amp = 0.5

    for phase in phases:
        # phase_corr = correct_pulse_params_by_phase(phase)
        # corrected_phase = np.round(phase + phase_corr)
        # corrected_phase = corrected_phase % 360
        # i_comp = np.cos(np.deg2rad(corrected_phase)) * amp
        # q_comp = np.sin(np.deg2rad(corrected_phase)) * amp
        # print(f"phase (deg): {(corrected_phase):.2f}")
        i_comp = np.cos(np.deg2rad(phase)) * amp
        q_comp = np.sin(np.deg2rad(phase)) * amp
        opx_config["waveforms"][f"i_{phase}"] = {"type": "constant", "sample": i_comp}
        opx_config["waveforms"][f"q_{phase}"] = {"type": "constant", "sample": q_comp}

    # Define the pulses and add the pulses to the elements
    for comp in ["i", "q"]:
        for pulse_name in pulse_names:
            for phase in phases:
                for chan in range(num_sig_gens):
                    # Define the pulse
                    full_pulse_name = f"ao_{comp}_{pulse_name}_{phase}_{chan}"
                    # print(full_pulse_name)
                    length = opx_config["pulses"][f"do_{pulse_name}_{chan}"]["length"] + 2*iq_buffer
                    # print(length)
                    opx_config["pulses"][full_pulse_name] = {
                        "operation": "control",
                        "length": length,
                        "waveforms": {"single": f"{comp}_{phase}"},
                    }
                    # Add the pulse to the element
                    dev = virtual_sig_gens_dict[chan]["physical_name"]
                    opx_config["elements"][f"ao_{dev}_{comp}"]["operations"][
                        f"{pulse_name}_{phase}"
                    ] = full_pulse_name

def build_phase_sweep(min_deg=-360, max_deg=360, step_deg=9):
    phases_unwrapped = np.arange(min_deg, max_deg + 1e-9, step_deg, dtype=int)  # -360..360
    phases_cmd = (phases_unwrapped % 360)  # wrap to [0,360)
    return phases_unwrapped.tolist(), phases_cmd.tolist()



# ref_img_array = np.array([])
# generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], [0, 90, 180, 270])
# fmt: off
# phases =[0, 45, 90, 135, 180, 225, 270, 315]
phases = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 
          99, 108, 117, 126, 135, 144, 153, 162, 171, 180,
          189, 198, 207, 216, 225, 234, 243, 252, 261, 270,
          279, 288, 297, 306, 315, 324, 333, 342, 351, 360]
# phases = [-360, -351, -342, -333, -324, -315, -306, -297, -288, -279, -270, -261, -252, -243, -234, -225, -216, -207, -198, -189, -180, -171, -162, -153, -144, -135, -126, -117, -108, -99, -90, -81, -72, -63, -54, -45, -36, -27, -18, -9, 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180, 189, 198, 207, 216, 225, 234, 243, 252, 261, 270, 279, 288, 297, 306, 315, 324, 333, 342, 351, 360]
# fmt:on
generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], phases)

if __name__ == "__main__":
    key = "pixel_to_sample_affine_transformation_matrix"
    mat = np.array(config["Positioning"][key])
    mat[:, 2] = [0, 0]
    print(mat)
    # Exactly what you want:
    phases_unwrapped, phases_cmd = build_phase_sweep(-360, 360, 9)
    print(phases_unwrapped)
    # generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], [0, 90])
