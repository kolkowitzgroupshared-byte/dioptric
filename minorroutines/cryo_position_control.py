import copy
import time
import labrad
import numpy as np
from utils import positioning as pos
from utils import kplotlib as kpl
# import majorroutines.confocal.determine_standard_readout_params as determine_standard_readout_params
# import majorroutines.confocal.g2_measurement as g2_measurement
import majorroutines.confocal.confocal_image_sample as image_sample
# import majorroutines.confocal.image_sample as image_sample

# import majorroutines.confocal.optimize_magnet_angle as optimize_magnet_angle
# import majorroutines.confocal.pulsed_resonance as pulsed_resonance
# import majorroutines.confocal.confocal_rabi as rabi

# import majorroutines.confocal.ramsey as ramsey
# import majorroutines.confocal.resonance as resonance
# import majorroutines.confocal.spin_echo as spin_echo
import majorroutines.confocal.confocal_stationary_count as stationary_count

# import majorroutines.confocal.t1_dq_main as t1_dq_main
# import majorroutines.confocal.targeting as targeting
import utils.tool_belt as tool_belt
from utils.constants import Axes, CoordsKey, NVSig, VirtualLaserKey

def get_sample_name() -> str:
    sample = "wu" #rubin
    return sample

sample_xy = [0,0] # piezo XY voltage input (1.0=1V) (not coordinates, relative)
coord_z = 0 # piezo z voltage (0 is the set midpoint, absolute) (negative is closer to smaple, move unit steps in sample; 37 is good surface focus with bs for Lovelace; 20 is good for dye)
pixel_xy = [0.0,0.0]  # sgalvso ref #potential NV?: [0.123, 0.139]

nv_sig = NVSig(
    name=f"({get_sample_name()})",
    coords={
        CoordsKey.SAMPLE: sample_xy,
        CoordsKey.Z: coord_z,
        CoordsKey.PIXEL: pixel_xy, #galvo 
    },
    disable_opt=False,
    disable_z_opt=True,
    expected_counts=13,
    pulse_durations={
        VirtualLaserKey.IMAGING: int(10e6), #10ms readout
        VirtualLaserKey.CHARGE_POL: int(1e4),
        VirtualLaserKey.SPIN_POL: 2000,
        VirtualLaserKey.SINGLET_DRIVE: 300,  # placeholder
    },
)
pos.set_xyz_on_nv(nv_sig)