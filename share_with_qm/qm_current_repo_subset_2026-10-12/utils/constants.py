# -*- coding: utf-8 -*-
"""Enums, dataclasses, other constants. Should not import anything other user modules
or else we will probably get a circular import

Created on June 26th, 2023

@author: mccambria
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto

from strenum import StrEnum

number = int | float


# Virtual laser keys are the names of virtual lasers, which accomplish one and only
### NOTE: 
class VirtualLaserKey(StrEnum):
    IMAGING = "imaging"
    RED_IMAGING = "red_imaging"
    ION = "ion"
    SCC = "scc"
    CHARGE_POL = "charge_pol"
    SPIN_POL = "spin_pol"
    SHELVING = "shelving"
    SPIN_READOUT = "spin_readout"
    WIDEFIELD_SHELVING = "widefield_shelving"
    WIDEFIELD_IMAGING = "widefield_imaging"
    WIDEFIELD_CHARGE_READOUT = "widefield_charge_readout"
    WIDEFIELD_SPIN_POL = "widefield_spin_pol"
    SINGLET_DRIVE = "singlet_drive"

# Coords keys are the names associated with the various physical coordinates
# for each NV. Each positioner name is a coords key. CoordsKey.PIXEL is associated
# with the location of the NV on a camera's pixel grid.
class CoordsKey(StrEnum):
    SAMPLE = "sample"
    PIXEL = "pixel"
    Z = "z"

@dataclass
class NVSig:
    name: str = None
    coords: dict[CoordsKey | str, list[float]] | list = None
    representative: bool = False
    disable_opt: bool = False
    disable_z_opt: bool = False
    threshold: number = None
    expected_counts: number = None
    magnet_angle: number = None
    opti_offset: list[number] = None  # Only works for global coordinates
    # spin_flip: If True, an additional pi pulse will be applied to the NV at
    # the end of a spin experiment prior to readout. Useful for anticorrelations
    # and rejecting common mode noise
    spin_flip: bool = False
    pulse_durations: dict[VirtualLaserKey, int] = field(default_factory=dict)
    pulse_amps: dict[VirtualLaserKey, float] = field(default_factory=dict)
    # nvn_dist_params: [bg, amp, sigma] for maximum likelihood state estimation
    nvn_dist_params: tuple = None


class CollectionMode(Enum):
    COUNTER = auto()  # Count all photons incident on a detector (e.g. APD)
    CAMERA = auto()  # Collect photons onto a camera


class ChargeStateEstimationMode(Enum):
    THRESHOLDING = auto()
    MLE = auto()  # Maximum likelihood estimator for images


class CountFormat(Enum):
    """Deprecated, everything should be raw"""

    KCPS = auto()  # Count rate in kilo counts per second
    RAW = auto()  # Just the raw number of counts


class Axes(Enum):
    NONE = ()
    X = (0,)
    Y = (1,)
    Z = (2,)
    XY = (0, 1)
    XYZ = (0, 1, 2)


class PosControlMode(Enum):
    """
    Different ways to control a positioner
    STEP: Manual control with individual function calls
    STREAM: A stream of values can be loaded onto the controller - the controller will step
        through the stream automatically in response to a clock signal from the pulse generator
    SEQUENCE: Controlled  directly from the pulse generator sequence
    """

    STEP = auto()
    STREAM = auto()
    SEQUENCE = auto()


class NVSpinState(Enum):
    LOW = auto()
    ZERO = auto()
    HIGH = auto()


# Normalization mode for comparing experimental data to reference data
class NormMode(Enum):
    SINGLE_VALUED = auto()  # Use a single-valued reference
    POINT_TO_POINT = auto()  # Normalize each signal point by its own reference


class ModMode(Enum):
    DIGITAL = auto()
    ANALOG = auto()


class Digital(IntEnum):
    LOW = 0
    HIGH = 1


Boltzmann = 8.617e-2  # meV / K


if __name__ == "__main__":
    test = NVSig(name="test")
    print(test["name"])
