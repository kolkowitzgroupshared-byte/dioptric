# -*- coding: utf-8 -*-s
"""
Config file for the PC cryo

Created Oct 7th, 2025
@author: chemistatcode
@author: sbchand
@author: ericvin
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

green_laser = "laser_COBO_520"  # make labrad server for COBOLT green laser
tisapph_laser = "laser_TISAPPH"  # fill this in later (labrad server for Tisapph)
thor_galvos = "pos_xy_THOR_gvs212"
cryo_piezo = "pos_xyz_ATTO_piezos"

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
    "collection_mode": CollectionMode.COUNTER,  # remove this line when set up in new computer
    # "charge_state_estimation_mode": ChargeStateEstimationMode.MLE,
    "charge_state_estimation_mode": ChargeStateEstimationMode.THRESHOLDING,
    "windows_repo_path": home / "GitHub/dioptric",
    "disable_z_drift_compensation": False,
    ###
    # Common durations are in ns
    ###
    "CommonDurations": {
        "cw_meas_buffer": 1000,
        "pol_to_uwave_wait_dur": 1000,
        "scc_ion_readout_buffer": 1000,
        "uwave_buffer": 100,
        "uwave_to_readout_wait_dur": 1000,
    },
    ###
    "DeviceIDs": {
        "arb_wave_gen_visa_address": "TCPIP0::128.104.ramp_to_zero_duration.119::5025::SOCKET",
        "daq0_name": "Dev1",
        "filter_slider_THOR_ell9k_com": "COM8",
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
        "sig_gen_STAN_sg394_visa": "TCPIP::192.168.0.120::inst0::INSTR",
        "sig_gen_STAN_sg394_2_visa": "TCPIP::192.168.0.121::inst0::INSTR",
        "sig_gen_STAN_sg394_3_visa": "TCPIP::192.168.0.177::inst0::INSTR",
        "sig_gen_TEKT_tsg4104a_visa": "TCPIP0::128.104.ramp_to_zero_duration.112::5025::SOCKET",
        "tagger_SWAB_20_1_serial": "1948000SIP", # cryo
        # "tagger_SWAB_20_1_serial": "1740000JEH", # nuclear
        "QM_opx_args": {
            "host": "192.168.0.117",
            "port": 9510,
            "cluster_name": "kolkowitz_nv_lab",
        },
        "tisapph_M2_solstis_ip": "192.168.0.195",
        "power_supply_RNS_ngc103_visa": "TCPIP::192.168.0.130::INSTR",
        "pos_xyz_ATTO_piezos_ip": "192.168.0.199",
        "filter_slider_THOR_ell9k_com": "COM5",
        "multimeter_KEIT_daq6510_visa": "TCPIP::192.168.0.122::inst0::INSTR",
        "laser_COBO_520_com": "COM4",

    },
    ###
    "Microwaves": {
        "PhysicalSigGens": {
            "sig_gen_BERK_bnc835": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394": {"delay": 104, "fm_mod_bandwidth": 100000.0},
            "sig_gen_STAN_sg394_3": {"delay": 151, "fm_mod_bandwidth": 100000.0},
            "sig_gen_TEKT_tsg4104a": {"delay": 57},
        },
        "iq_comp_amp": 0.5,
        "iq_delay": 140,
        "VirtualSigGens": {
            0: {
                "physical_name": "sig_gen_STAN_sg394_3",
                "uwave_power": 10, #dbm
                "frequency": 2.8214, #GHz
                "rabi_period": 184.9,
                "pi_pulse": 92.4,
                "pi_on_2_pulse": 64, #Half of pi pulse, for use in Ramsey and SE
            },
            # sig gen 1 is iq molulated
            1: {
                "physical_name": "sig_gen_STAN_sg394_4",
                "uwave_power": 6.0,
                "frequency": 2.8360,
                "rabi_period": 144,
                "pi_pulse": 72,
                "pi_on_2_pulse": 36,
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
                # "delay": 0,
                "delay": 960, #960ns, Characterized by Saroj and Caitlin on 04/17/2025
                "mod_mode": ModMode.DIGITAL,
                "positioner": CoordsKey.PIXEL,
            },
        },
        "VirtualLasers": {
            # LaserKey.IMAGING: {"physical_name": green_laser, "duration": 50e6},
            VirtualLaserKey.IMAGING: {
                # "physical_name": green_laser,
                "physical_name": green_laser,  # this is the laser that appears on the imaging APD scan
                "duration": 10e6,  # this duration appears on the imaging APD scan
            },
            VirtualLaserKey.SINGLET_DRIVE: {
                "physical_name": tisapph_laser,
                "duration": 100e3,  # this is a placeholder
            },

            VirtualLaserKey.SPIN_READOUT: {
                "physical_name": green_laser,
                "duration": 440,
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 10e3},
            VirtualLaserKey.CHARGE_POL: {
                "physical_name": green_laser,
                "duration": 1e3,  # Works better for Deep NVs (Johnson)
            },
            # LaserKey.CHARGE_POL: {"physical_name": green_laser, "duration": 60},
            VirtualLaserKey.SPIN_POL: {
                "physical_name": green_laser,
                "duration": 2e3,
            },
            VirtualLaserKey.SHELVING: {
                "physical_name": green_laser,
                "duration": 60,
            },

        },
        #
        "PulseSettings": {
            "scc_shelving_pulse": False,  # Example setting
        },  # Whether or not to include a shelving pulse in SCC
    },
    ###
    "Positioning": {
        "drift_xy_coords_key": CoordsKey.PIXEL,
        "Positioners": {
            # update with correct piezos for cryo
            CoordsKey.SAMPLE: {
                "physical_name": "pos_xyz_ATTO_piezos", #xy atto
                # "control_mode": PosControlMode.STREAM,
                "control_mode": PosControlMode.STEP,
                "delay": int(1e6),  # 5 ms for PIFOC xyz
                "nm_per_unit": 1000,
                "optimize_range": 0.1,
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
            CoordsKey.Z: {
                # "physical_name": "pos_xyz_ATTO_piezos", #z atto
                "physical_name": "pos_z_PI_pifoc", #z atto
                "control_mode": PosControlMode.STEP,
                # "delay": int(1e6),  # 1 ms for ATTO
                "delay": int(5e6),  # 5 ms for PIFOC xyz
                "nm_per_unit": 1000,
                "optimize_range": 0.1,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
            CoordsKey.PIXEL: {
                "physical_name": "pos_xy_THOR_gvs212",
                "control_mode": PosControlMode.STEP,
                "delay": int(400e3),  # 400 us for galvo
                "nm_per_unit": 1000,
                "optimize_range": 0.01,
                "units": "Voltage (V)",
                "opti_virtual_laser_key": VirtualLaserKey.IMAGING,
            },
        },
        "pixel_to_sample_affine_transformation_matrix": pixel_to_sample_affine_transformation_matrix,
        "cryo_piezos_voltage": 33,
        "z_bias_adjust": 0.0,
        "optimize_num_steps": 20,
    },
    ###
    "Servers": {  # Bucket for miscellaneous servers not otherwise listed above
        "pulse_streamer": "pulse_gen_SWAB_82",
        "counter": "tagger_SWAB_20",
    },
    ###
    "Wiring": {
        "Daq": {
            # https://docs-be.ni.com/bundle/ni-67xx-scb-68a-labels/raw/resource/enus/371806a.pdf
            "ao_galvo_x": "dev1/ao11",
            "ao_galvo_y": "dev1/ao4",
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
            # clocks / gates
            "do_sample_clock": 0,  # 125 MHz-compatible sample clock out to Tagger
            "do_apd_gate": 1,  # gate line to Tagger
            # "do_camera_trigger": 6,  # optional
            # "do_laser_INTE_520_dm": 2,  # green  TTL
            "do_laser_COBO_520_dm": 2,
            # "do_laser_COBO_638_dm": 3,  # red TTL
            # microwaves (TTL gate to SGs)
            "do_sig_gen_STAN_sg394_3_dm": 4,
            # "do_sig_gen_STAN_sg394_dm": 5,
            # analog (for the yellow AOM amplitude)
            "ao_laser_OPTO_589_am": 0,  # yellow analog modulation
            "do_laser_TISAPPH_dm":3,  # Tisapph TTL modulation
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

if __name__ == "__main__":
    key = "pixel_to_sample_affine_transformation_matrix"
    mat = np.array(config["Positioning"][key])
    mat[:, 2] = [0, 0]
    print(mat)
    # generate_iq_pulses(["pi_pulse", "pi_on_2_pulse"], [0, 90])
