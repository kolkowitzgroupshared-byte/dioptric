"""Minimal Python wrapper for the BBS DLP6500 ALC controller.

This module wraps the vendor DLL/.so shipped in the user's SDK package.
It is designed for laboratory integration where masks are uploaded over USB,
then selected deterministically using the controller's sequence engine and
external trigger inputs.

Tested only at the API-signature level against the shipped headers/samples.
Hardware-specific timing and trigger polarity should be validated on the bench.
"""

from __future__ import annotations

import ctypes
import os
import sys
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from scipy.optimize import curve_fit
from skimage.feature import blob_log
from skimage.filters import gaussian
from skimage.draw import disk

# Geometry from vendor docs
DMD_WIDTH = 1920
DMD_HEIGHT = 1080
PLANE_PADDED_WIDTH = 2048  # plane buffer is padded to 2048 bits wide
PLANE_BYTES = PLANE_PADDED_WIDTH * DMD_HEIGHT // 8
SEQ_BUFFER_SIZE = 131072
IDLE_SEQUENCE_START = SEQ_BUFFER_SIZE

# Sequence / command constants from API_defines.h
CMD_NOP = 0x10000000
CMD_OUTPUT = 0x80000000
CMD_GLOB_MIRRORCLOCKING = 0x40000000
CMD_GLOB_CLEAR = 0xE0000000
CMD_BLOCK_CLEAR = 0x30000000
CMD_FLOAT = 0x50000000
CMD_JUMP_TO = 0xF4000000
CMD_JUMP_RELATIVE = 0xF3000000
CMD_CALL = 0xF5000000
CMD_RETURN = 0xFE000000
CMD_SEQ_END = 0xFF000000
CMD_WAIT_US_SINCE_MCP = 0xF8000000
CMD_WAIT_FOR_EVENT = 0xF9000000
CMD_CLEAR_EVENT = 0xBA000000
CMD_TIMERSTART = 0xA0000000
CMD_IF_EVENT = 0xB8000000
CMD_IF_LATCHED_EVENT = 0xB9000000
CMD_GLOB_LOAD = 0xC0000000
CMD_REGSET = 0xD0000000

COND_ALWAYS = 0
COND_IF_TRUE = 0x00100000
COND_IF_FALSE = 0x00200000
JUMP_FORWARD = 0
JUMP_BACKWARD = 0x00100000

LOAD_PLANE_NR = 0x00000000
LOAD_REG0_PLUS_REG1 = 0x01000000
LOAD_REG0_PLUS_REG2 = 0x02000000
LOAD_FROM_LINENR_IN_REG0 = 0x04000000
LOAD_FROM_LINENR = 0x07000000
LOAD_OFFSET_REG0 = 0x08000000
LOAD_OFFSET_REG1 = 0x09000000
LOAD_OFFSET_REG2 = 0x0A000000
LOAD_OFFSET_REG3 = 0x0B000000
LOAD_OFFSET_REG4 = 0x0C000000
LOAD_OFFSET_REG5 = 0x0D000000
LOAD_OFFSET_REG6 = 0x0E000000
LOAD_OFFSET_REG7 = 0x0F000000

REG0 = 0
REG1 = 1

MODE_FLIP_X = 1 << 0
MODE_FLIP_Y = 1 << 1
MODE_COMPLEMENT = 1 << 2
MODE_TEMP_OVERRIDE = 1 << 7
WDT_DISABLE = 1 << 16
RESET2BLK_Z = 1 << 17

EVENT_TRIG0_POSLEV = 0x00000001
EVENT_TRIG1_POSLEV = 0x00000002
EVENT_TRIG0_NEGLEV = 0x00000004
EVENT_TRIG1_NEGLEV = 0x00000008
EVENT_TRIG0_POSEDGE = 0x00000010
EVENT_TRIG1_POSEDGE = 0x00000020
EVENT_TRIG0_NEGEDGE = 0x00000040
EVENT_TRIG1_NEGEDGE = 0x00000080
EVENT_EVENT_SOFT = 0x00000100
EVENT_EVENT_HDMI = 0x00000200
EVENT_EVENT_USB = 0x00000400
EVENT_ALL = 0x0000FFFF

OUT_PIN0 = 0x01000000
OUT_PIN1 = 0x02000000
OUT_DMD_FLAGS_PERM = 0x04000000
OUT_DMD_FLAGS_TEMP = 0x05000000


class DmdError(RuntimeError):
    pass


@dataclass(frozen=True)
class TriggerPair:
    on_event: int
    off_event: int


TRIG0_LEVEL = TriggerPair(EVENT_TRIG0_POSLEV, EVENT_TRIG0_NEGLEV)
TRIG1_LEVEL = TriggerPair(EVENT_TRIG1_POSLEV, EVENT_TRIG1_NEGLEV)
TRIG0_EDGE = TriggerPair(EVENT_TRIG0_POSEDGE, EVENT_TRIG0_NEGEDGE)
TRIG1_EDGE = TriggerPair(EVENT_TRIG1_POSEDGE, EVENT_TRIG1_NEGEDGE)


class Dmd6500:
    def __init__(self, library_path: str | os.PathLike[str]):
        self.library_path = str(library_path)
        self.lib = self._load_library(self.library_path)
        self._bind_api()
        self.handle: int | None = None
        self.device_id: int | None = None

    @staticmethod
    def _load_library(library_path: str):
        system = platform.system().lower()
        if system.startswith("win"):
            return ctypes.WinDLL(library_path)
        return ctypes.CDLL(library_path)

    def _bind_api(self) -> None:
        # device management
        self.lib.ListControllers.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self.lib.ListControllers.restype = ctypes.c_int

        self.lib.GetDevID.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
        self.lib.GetDevID.restype = ctypes.c_int

        self.lib.GetSerialNumber.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.GetSerialNumber.restype = ctypes.c_int

        self.lib.GetDevice.argtypes = []
        self.lib.GetDevice.restype = ctypes.c_void_p

        self.lib.DeleteDevice.argtypes = [ctypes.c_void_p]
        self.lib.DeleteDevice.restype = None

        self.lib.Connect.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.Connect.restype = ctypes.c_int

        self.lib.Disconnect.argtypes = [ctypes.c_void_p]
        self.lib.Disconnect.restype = ctypes.c_int

        # transfer / display
        self.lib.SendImageMono.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.lib.SendImageMono.restype = ctypes.c_int

        self.lib.SendPlane.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.lib.SendPlane.restype = ctypes.c_int

        self.lib.CalcPlane.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
        ]
        self.lib.CalcPlane.restype = None

        self.lib.LoadPlaneToDLP.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.LoadPlaneToDLP.restype = ctypes.c_int

        self.lib.DLP_GlobalMCP.argtypes = [ctypes.c_void_p]
        self.lib.DLP_GlobalMCP.restype = ctypes.c_int

        self.lib.WriteCommand.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.lib.WriteCommand.restype = ctypes.c_int

        self.lib.SendSequenceData.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.SendSequenceData.restype = ctypes.c_int

        self.lib.RunSequence.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.RunSequence.restype = ctypes.c_int

        self.lib.StopSequence.argtypes = [ctypes.c_void_p]
        self.lib.StopSequence.restype = ctypes.c_int

        self.lib.GetFirmwareVersion.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.lib.GetFirmwareVersion.restype = ctypes.c_int

    def _require_handle(self) -> ctypes.c_void_p:
        if self.handle is None:
            raise DmdError("Device not connected")
        return ctypes.c_void_p(self.handle)

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise DmdError(f"{what} failed with code {rc}")

    def list_devices(self) -> int:
        count = ctypes.c_uint(0)
        self._check(self.lib.ListControllers(ctypes.byref(count)), "ListControllers")
        return int(count.value)

    def get_device_id(self, index: int = 0) -> int:
        dev_id = ctypes.c_int(0)
        self._check(self.lib.GetDevID(index, ctypes.byref(dev_id)), "GetDevID")
        return int(dev_id.value)

    def get_serial(self, device_id: int) -> str:
        buf = ctypes.create_string_buffer(64)
        self._check(self.lib.GetSerialNumber(device_id, buf), "GetSerialNumber")
        return buf.value.decode(errors="replace")

    def connect(self, index: int = 0) -> None:
        if self.handle is not None:
            return
        if self.list_devices() <= index:
            raise DmdError(f"Requested device index {index}, but not enough controllers found")
        self.device_id = self.get_device_id(index)
        handle = self.lib.GetDevice()
        if not handle:
            raise DmdError("GetDevice returned null")
        self._check(self.lib.Connect(handle, self.device_id), "Connect")
        self.handle = int(handle)

    def disconnect(self) -> None:
        if self.handle is None:
            return
        handle = ctypes.c_void_p(self.handle)
        try:
            self.lib.Disconnect(handle)
        finally:
            self.lib.DeleteDevice(handle)
            self.handle = None
            self.device_id = None

    def firmware_version(self) -> int:
        version = ctypes.c_uint(0)
        self._check(self.lib.GetFirmwareVersion(self._require_handle(), ctypes.byref(version)), "GetFirmwareVersion")
        return int(version.value)

    @staticmethod
    def _as_gray_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.shape != (DMD_HEIGHT, DMD_WIDTH):
            raise ValueError(f"Expected image shape {(DMD_HEIGHT, DMD_WIDTH)}, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return np.ascontiguousarray(arr)

    @staticmethod
    def binary_mask_from_spots(
        spots_xy: Sequence[tuple[float, float]],
        radius_px: int = 8,
        invert: bool = False,
    ) -> np.ndarray:
        y, x = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
        mask = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
        for cx, cy in spots_xy:
            rr2 = (x - float(cx)) ** 2 + (y - float(cy)) ** 2
            mask[rr2 <= radius_px**2] = 255
        if invert:
            mask = 255 - mask
        return mask

    def send_image_mono(self, start_plane: int, image_u8: np.ndarray) -> None:
        """Upload one 8-bit grayscale image.

        The vendor API expands the 8-bit grayscale image into 8 consecutive bitplanes,
        starting at `start_plane`.
        """
        img = self._as_gray_image(image_u8)
        ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self._check(self.lib.SendImageMono(self._require_handle(), start_plane, ptr), "SendImageMono")

    def calc_binary_plane(self, image_u8: np.ndarray, bitlevel: int = 0) -> np.ndarray:
        gray = self._as_gray_image(image_u8).reshape(-1)
        plane = np.zeros(PLANE_BYTES, dtype=np.uint8)
        self.lib.CalcPlane(
            gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            plane.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            int(bitlevel),
        )
        return plane

    def send_binary_plane(self, plane_nr: int, binary_image: np.ndarray, bitlevel: int = 0) -> None:
        plane = self.calc_binary_plane(binary_image, bitlevel=bitlevel)
        ptr = plane.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self._check(self.lib.SendPlane(self._require_handle(), int(plane_nr), ptr), "SendPlane")

    def show_plane(self, plane_nr: int) -> None:
        handle = self._require_handle()
        self.stop_sequence()
        self._check(self.lib.LoadPlaneToDLP(handle, int(plane_nr)), "LoadPlaneToDLP")
        self._check(self.lib.DLP_GlobalMCP(handle), "DLP_GlobalMCP")

    def write_command(self, cmd: int) -> None:
        self._check(self.lib.WriteCommand(self._require_handle(), ctypes.c_uint(cmd)), "WriteCommand")

    def load_and_clock(self, plane_nr: int) -> None:
        self.write_command(CMD_GLOB_LOAD | LOAD_PLANE_NR | int(plane_nr))
        self.write_command(CMD_GLOB_MIRRORCLOCKING)

    def upload_sequence(self, commands: Iterable[int], startpos: int = 1000) -> np.ndarray:
        seq = np.asarray(list(commands), dtype=np.uint32)
        self._check(
            self.lib.SendSequenceData(
                self._require_handle(),
                seq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                0,
                int(seq.size),
                int(startpos),
            ),
            "SendSequenceData",
        )
        return seq

    def run_sequence(self, startpos: int) -> None:
        self._check(self.lib.RunSequence(self._require_handle(), int(startpos)), "RunSequence")

    def stop_sequence(self) -> None:
        self._check(self.lib.StopSequence(self._require_handle()), "StopSequence")

    def program_level_gate_sequence(
        self,
        on_plane: int,
        off_plane: int,
        trigger: TriggerPair = TRIG0_LEVEL,
        startpos: int = 1000,
    ) -> np.ndarray:
        """Display `on_plane` while the trigger is high, otherwise display `off_plane`."""
        cmds = [
            CMD_WAIT_FOR_EVENT | trigger.on_event,
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(on_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_FOR_EVENT | trigger.off_event,
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(off_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_JUMP_TO | int(startpos),
        ]
        return self.upload_sequence(cmds, startpos=startpos)

    def program_two_window_sequence(
        self,
        prep_plane: int,
        read_plane: int,
        off_plane: int,
        prep_trigger: TriggerPair = TRIG0_LEVEL,
        read_trigger: TriggerPair = TRIG1_LEVEL,
        startpos: int = 1100,
    ) -> np.ndarray:
        """Useful for separate charge-prep and readout windows controlled by two trigger lines."""
        cmds = [
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(off_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_FOR_EVENT | prep_trigger.on_event,
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(prep_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_FOR_EVENT | prep_trigger.off_event,
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(off_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_FOR_EVENT | read_trigger.on_event,
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(read_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_FOR_EVENT | read_trigger.off_event,
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(off_plane),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_JUMP_TO | int(startpos),
        ]
        return self.upload_sequence(cmds, startpos=startpos)

    def program_blink_sequence(
        self,
        plane_a: int,
        plane_b: int,
        dwell_us: int = 500_000,
        startpos: int = 1200,
    ) -> np.ndarray:
        """
        Alternate between plane_a and plane_b with a fixed dwell time.
        No external trigger required.
        """
        if dwell_us <= 0:
            raise ValueError("dwell_us must be > 0")

        cmds = [
            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(plane_a),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_US_SINCE_MCP | int(dwell_us),

            CMD_GLOB_LOAD | LOAD_PLANE_NR | int(plane_b),
            CMD_GLOB_MIRRORCLOCKING,
            CMD_WAIT_US_SINCE_MCP | int(dwell_us),

            CMD_JUMP_TO | int(startpos),
        ]
        return self.upload_sequence(cmds, startpos=startpos)



def default_library_path(sdk_root: str | os.PathLike[str]) -> Path:
    sdk_root = Path(sdk_root)
    system = platform.system().lower()
    if system.startswith("win"):
        return sdk_root / "Windows_x86_64" / "DMD6500_GUI" / "DLP6500_DLL.dll"
    return sdk_root / "Linux_x86_64" / "Linux_API" / "libbbs_api.so"

def bullseye_crosshair_pattern():
    y, x = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    cx, cy = DMD_WIDTH // 2, DMD_HEIGHT // 2

    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)

    # Bullseye rings
    for r0, r1 in [(60, 90), (140, 170), (240, 270), (360, 390)]:
        img[(r >= r0) & (r <= r1)] = 255

    # Crosshair
    img[:, cx-6:cx+6] = 255
    img[cy-6:cy+6, :] = 255

    # Four corner fiducials
    for px, py in [(150, 150), (DMD_WIDTH-150, 150), (150, DMD_HEIGHT-150), (DMD_WIDTH-150, DMD_HEIGHT-150)]:
        rr2 = (x - px)**2 + (y - py)**2
        img[rr2 <= 35**2] = 255

    return img


def smiley_pattern():
    y, x = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    cx, cy = DMD_WIDTH // 2, DMD_HEIGHT // 2

    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)

    # Face outline
    rr2 = (x - cx)**2 + (y - cy)**2
    img[(rr2 >= 260**2) & (rr2 <= 300**2)] = 255

    # Eyes
    for ex in [cx - 110, cx + 110]:
        eye = (x - ex)**2 + (y - (cy - 80))**2
        img[eye <= 28**2] = 255

    # Smile arc
    r = np.sqrt((x - cx)**2 + (y - (cy + 20))**2)
    theta = np.arctan2(y - (cy + 20), x - cx)
    smile = (r >= 150) & (r <= 180) & (theta > 0.25) & (theta < 2.89)
    img[smile] = 255

    return img


def pinwheel_pattern():
    y, x = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    cx, cy = DMD_WIDTH // 2, DMD_HEIGHT // 2

    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)

    angle = np.arctan2(y - cy, x - cx)
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)

    # 8 angular sectors, every other one bright
    sectors = ((angle + np.pi) / (np.pi / 4)).astype(int) % 2
    img[(sectors == 0) & (radius < 420)] = 255

    # Add a hollow center
    img[radius < 80] = 0

    return img

def checkerboard_pattern(block=80):
    y, x = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    img = (((x // block) + (y // block)) % 2).astype(np.uint8) * 255
    return img

def corner_dots_pattern():
    y, x = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    for px, py in [(120, 120), (DMD_WIDTH-120, 120), (120, DMD_HEIGHT-120), (DMD_WIDTH-120, DMD_HEIGHT-120)]:
        rr2 = (x - px)**2 + (y - py)**2
        img[rr2 <= 40**2] = 255
    return img


import numpy as np

def all_off_pattern():
    return np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)

def all_on_pattern():
    return np.full((DMD_HEIGHT, DMD_WIDTH), 255, dtype=np.uint8)

def left_half_pattern():
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    img[:, :DMD_WIDTH // 2] = 255
    return img

def right_half_pattern():
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    img[:, DMD_WIDTH // 2:] = 255
    return img

def vertical_stripe_pattern(width=250):
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    cx = DMD_WIDTH // 2
    img[:, cx - width // 2: cx + width // 2] = 255
    return img

def horizontal_stripe_pattern(height=250):
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    cy = DMD_HEIGHT // 2
    img[cy - height // 2: cy + height // 2, :] = 255
    return img

def center_square_pattern(size=500):
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    cx = DMD_WIDTH // 2
    cy = DMD_HEIGHT // 2
    x0 = max(0, cx - size // 2)
    x1 = min(DMD_WIDTH, cx + size // 2)
    y0 = max(0, cy - size // 2)
    y1 = min(DMD_HEIGHT, cy + size // 2)
    img[y0:y1, x0:x1] = 255
    return img


# if __name__ == "__main__":
#     lib_path = r"C:\Users\jkdol\OneDrive\Documents\Github\dioptric\dmdsuite\Windows_x86_64\DLL_x64\x64\Release\DLP6500_DLL.dll"

#     dmd = Dmd6500(lib_path)
#     print(dmd.list_devices(), "device(s) found")

#     try:
#         dmd.connect(0)
#         dmd.send_binary_plane(200, all_off_pattern())
#         dmd.send_binary_plane(201, all_on_pattern())
#         dmd.send_binary_plane(202, left_half_pattern())
#         dmd.send_binary_plane(203, vertical_stripe_pattern())
#         dmd.send_binary_plane(204, horizontal_stripe_pattern())
#         dmd.send_binary_plane(205, center_square_pattern())


#         for plane, name in [
#             (200, "OFF"),
#             (201, "ALL ON"),
#             (202, "LEFT HALF"),
#             (203, "VERTICAL STRIPE"),
#             (204, "HORIZONTAL STRIPE"),
#             (205, "CENTER SQUARE"),
#             (200, "OFF"),
#         ]:
#             print(f"Showing {name}")
#             dmd.show_plane(plane)
#             input(f"Inspect {name}, then press Enter...")

#     finally:
#         dmd.disconnect()
# sys.exit()

# ============================================================
# OFF-state-pass DMD calibration:
#   1. block 0th order
#   2. triangle calibration -> camera-to-DMD affine
#   3. circle movie using affine mapping
# ============================================================

import os
import time
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Basic DMD states for YOUR current alignment
# ============================================================

def dmd_pass_all_pattern():
    """
    Your current alignment:
        black / 0 / DMD OFF = pass to camera.
    """
    return np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)


def dmd_block_all_pattern():
    """
    Your current alignment:
        white / 255 / DMD ON = deflect/block from camera.
    """
    return np.full((DMD_HEIGHT, DMD_WIDTH), 255, dtype=np.uint8)


def all_off_pattern():
    # Kept for compatibility.
    # In your current alignment this is PASS ALL.
    return dmd_pass_all_pattern()


def all_on_pattern():
    # Kept for compatibility.
    # In your current alignment this is BLOCK ALL.
    return dmd_block_all_pattern()


# ============================================================
# Camera helpers
# ============================================================

def safe_get_image(cam, exposure=0.0001, tries=20, delay_s=0.05):
    """
    Robust camera grab. ThorCam sometimes returns None on first few calls.
    """
    cam.set_exposure(exposure)
    time.sleep(0.12)

    for _ in range(tries):
        img = cam.get_image()
        if img is not None:
            return img
        time.sleep(delay_s)

    raise RuntimeError("Camera returned None after multiple attempts.")


def brightest_spot_centroid(img, threshold_percentile=99.8, plot=True):
    """
    Find brightest spot using connected components, then return
    intensity-weighted centroid instead of binary-mask centroid.
    """
    imgf = np.asarray(img).astype(np.float32)

    thresh = np.percentile(imgf, threshold_percentile)
    mask = (imgf >= thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    best_i = None
    best_sum = -np.inf

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 3 <= area <= 8000:
            total = imgf[labels == i].sum()
            if total > best_sum:
                best_sum = total
                best_i = i

    if best_i is None:
        raise RuntimeError("Could not find brightest spot / 0th order.")

    # Pixels belonging to the brightest connected component
    ys, xs = np.where(labels == best_i)
    weights = imgf[ys, xs]

    # Optional: subtract local/background threshold so halo contributes less
    weights = weights - thresh
    weights = np.clip(weights, 0, None)

    if weights.sum() <= 0:
        # fallback to binary centroid
        xy = centroids[best_i].astype(np.float32)
    else:
        x_c = np.sum(xs * weights) / np.sum(weights)
        y_c = np.sum(ys * weights) / np.sum(weights)
        xy = np.array([x_c, y_c], dtype=np.float32)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap="gray")
        plt.plot(xy[0], xy[1], "rx", markersize=12, label="weighted centroid")

        # also show brightest pixel for comparison
        y_peak, x_peak = np.unravel_index(np.argmax(imgf), imgf.shape)
        plt.title(f"0th order camera xy = {xy}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return xy


def detect_spot_centroids(
    img,
    threshold_percentile=99.5,
    min_area=3,
    max_area=3000,
    zero_order_xy=None,
    zero_order_exclusion_radius=50,
    plot=True,
    title="Detected spots",
):
    """
    Detect bright SLM spots in camera image.
    Returns Nx2 array of camera coordinates [x, y].
    """
    imgf = np.asarray(img).astype(np.float32)

    thresh = np.percentile(imgf, threshold_percentile)
    mask = (imgf >= thresh).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    pts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            x, y = centroids[i]

            if zero_order_xy is not None:
                zx, zy = zero_order_xy
                if np.hypot(x - zx, y - zy) < zero_order_exclusion_radius:
                    continue

            pts.append([x, y])

    pts = np.array(pts, dtype=np.float32)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap="gray")
        for k, p in enumerate(pts):
            plt.plot(p[0], p[1], "ro", markersize=2)
            plt.text(p[0] + 5, p[1] + 5, str(k), color="red")
        if zero_order_xy is not None:
            plt.plot(zero_order_xy[0], zero_order_xy[1], "gx", markersize=6)
        plt.title(f"{title}: {len(pts)} spots")
        plt.tight_layout()
        plt.show()

    return pts


def integrate_spot_intensities(img, spot_pts, roi=8):
    """
    Integrate intensity around camera spots.
    spot_pts are camera coordinates [x, y].
    """
    img = np.asarray(img).astype(np.float32)
    h, w = img.shape
    vals = []

    for x, y in spot_pts:
        x = int(round(x))
        y = int(round(y))

        x0 = max(0, x - roi)
        x1 = min(w, x + roi + 1)
        y0 = max(0, y - roi)
        y1 = min(h, y + roi + 1)

        vals.append(img[y0:y1, x0:x1].sum())

    return np.array(vals, dtype=np.float32)


# ============================================================
# DMD mask helpers for OFF-pass alignment
# ============================================================

def vertical_blocking_stripe_mask(cx, width=30):
    """
    Black everywhere = pass.
    White vertical stripe = block/deflect.
    """
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    x0 = max(0, int(cx - width // 2))
    x1 = min(DMD_WIDTH, int(cx + width // 2))
    img[:, x0:x1] = 255
    return img


def horizontal_blocking_stripe_mask(cy, height=30):
    """
    Black everywhere = pass.
    White horizontal stripe = block/deflect.
    """
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    y0 = max(0, int(cy - height // 2))
    y1 = min(DMD_HEIGHT, int(cy + height // 2))
    img[y0:y1, :] = 255
    return img


def block_spots_mask(dmd_pts, block_indices, radius_px=20):
    """
    PASS everywhere, BLOCK selected spots.
    black background = pass
    white circles = block
    """
    mask = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    yy, xx = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    pts = np.asarray(dmd_pts, dtype=np.float32)

    for idx in block_indices:
        x, y = pts[idx]
        rr2 = (xx - x) ** 2 + (yy - y) ** 2
        mask[rr2 <= radius_px**2] = 255

    return mask


def pass_selected_spots_mask(dmd_pts, pass_indices, radius_px=20):
    """
    BLOCK everywhere, PASS selected spots.
    white background = block
    black circles = pass
    """
    mask = np.full((DMD_HEIGHT, DMD_WIDTH), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    pts = np.asarray(dmd_pts, dtype=np.float32)

    for idx in pass_indices:
        x, y = pts[idx]
        rr2 = (xx - x) ** 2 + (yy - y) ** 2
        mask[rr2 <= radius_px**2] = 0

    return mask


def block_single_dmd_point_mask(cx, cy, radius_px=50):
    """
    PASS everywhere, BLOCK one DMD coordinate.
    Good for 0th-order blocking.
    """
    mask = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
    yy, xx = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[rr2 <= radius_px**2] = 255
    return mask


def apply_zero_block(mask, zero_dmd_xy, zero_radius_px=60):
    """
    Add permanent 0th-order blocking hole to any mask.
    Since white = block, this sets a white circle at zero_dmd_xy.
    """
    if zero_dmd_xy is None:
        return mask

    mask = mask.copy()
    yy, xx = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
    x, y = zero_dmd_xy
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    mask[rr2 <= zero_radius_px**2] = 255
    return mask


def sort_spots_clockwise(cam_pts):
    """
    Sort detected circle spots by angle around their camera-space center.
    """
    cam_pts = np.asarray(cam_pts, dtype=np.float32)
    center = cam_pts.mean(axis=0)
    angles = np.arctan2(cam_pts[:, 1] - center[1], cam_pts[:, 0] - center[0])
    return np.argsort(angles)


# ============================================================
# Blocking-drop scans
# ============================================================

def scan_dmd_axis_for_spots_blocking(
    dmd,
    cam,
    cam_pts,
    axis="x",
    positions=None,
    stripe_width=30,
    plane=220,
    pass_plane=200,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=60,
):
    """
    OFF-pass alignment calibration.

    Reference: black pass plane gives full intensity.
    Then a white blocking stripe is scanned.
    Correct coordinate gives maximum intensity DROP.
    """
    if positions is None:
        if axis == "x":
            positions = np.arange(900, 1600, 20)
        else:
            positions = np.arange(200, 750, 20)

    # Reference image: everything passes, except optional 0th-order block.
    dmd.show_plane(pass_plane)
    time.sleep(0.1)
    img_pass = safe_get_image(cam, exposure=exposure)
    pass_vals = integrate_spot_intensities(img_pass, cam_pts, roi=roi)

    drops = []

    for p in positions:
        if axis == "x":
            mask = vertical_blocking_stripe_mask(p, width=stripe_width)
        else:
            mask = horizontal_blocking_stripe_mask(p, height=stripe_width)

        mask = apply_zero_block(mask, zero_dmd_xy, zero_radius_px=zero_radius_px)

        dmd.send_binary_plane(plane, mask)
        dmd.show_plane(plane)

        time.sleep(0.06)
        img = safe_get_image(cam, exposure=exposure)
        vals = integrate_spot_intensities(img, cam_pts, roi=roi)

        drop = pass_vals - vals
        drop = np.clip(drop, 0, None)

        drops.append(drop)
        print(f"{axis} scan {p}: max drop {drop.max():.3g}", flush=True)

    drops = np.array(drops)
    best_indices = np.argmax(drops, axis=0)
    best_positions = np.asarray(positions)[best_indices]

    return np.asarray(positions), drops, best_positions


def plot_response_curves(positions, responses, spot_indices=(0, 1, 2), title=""):
    plt.figure(figsize=(8, 4))
    for i in spot_indices:
        if i < responses.shape[1]:
            plt.plot(positions, responses[:, i], "-o", label=f"spot {i}")
    plt.xlabel("DMD scan position")
    plt.ylabel("intensity drop")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_dmd_xy_for_camera_spots_blocking(
    dmd,
    cam,
    cam_pts,
    x_positions=np.arange(600, 1200, 20),
    y_positions=np.arange(200, 750, 20),
    stripe_width=30,
    x_plane=220,
    y_plane=221,
    pass_plane=200,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=60,
    plot=True,
):
    """
    Find DMD coordinates for camera spots by x/y blocking stripe scans.
    """
    x_positions, x_drop, dmd_x = scan_dmd_axis_for_spots_blocking(
        dmd=dmd,
        cam=cam,
        cam_pts=cam_pts,
        axis="x",
        positions=x_positions,
        stripe_width=stripe_width,
        plane=x_plane,
        pass_plane=pass_plane,
        exposure=exposure,
        roi=roi,
        zero_dmd_xy=zero_dmd_xy,
        zero_radius_px=zero_radius_px,
    )

    if plot:
        plot_response_curves(x_positions, x_drop, spot_indices=range(min(5, len(cam_pts))), title="DMD x drop scan")

    input("X drop scan done. Press Enter for Y drop scan...")

    y_positions, y_drop, dmd_y = scan_dmd_axis_for_spots_blocking(
        dmd=dmd,
        cam=cam,
        cam_pts=cam_pts,
        axis="y",
        positions=y_positions,
        stripe_width=stripe_width,
        plane=y_plane,
        pass_plane=pass_plane,
        exposure=exposure,
        roi=roi,
        zero_dmd_xy=zero_dmd_xy,
        zero_radius_px=zero_radius_px,
    )

    if plot:
        plot_response_curves(y_positions, y_drop, spot_indices=range(min(5, len(cam_pts))), title="DMD y drop scan")

    dmd_pts = np.column_stack([dmd_x, dmd_y]).astype(np.float32)

    return dmd_pts, x_positions, y_positions, x_drop, y_drop


# ============================================================
# Local refinement in OFF-pass mode
# ============================================================

def refine_dmd_point_for_spot_blocking(
    dmd,
    cam,
    cam_pts,
    spot_index,
    initial_xy,
    search_radius=30,
    step=10,
    aperture_radius=15,
    plane=240,
    pass_plane=200,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=60,
    leakage_weight=0.5,
):
    """
    Refine DMD coordinate by scanning a small WHITE blocking disk.
    Score = target drop - leakage_weight * strongest other drop.
    """
    x0, y0 = initial_xy

    xs = np.arange(x0 - search_radius, x0 + search_radius + 1, step)
    ys = np.arange(y0 - search_radius, y0 + search_radius + 1, step)

    # Reference pass image.
    dmd.show_plane(pass_plane)
    time.sleep(0.1)
    img_pass = safe_get_image(cam, exposure=exposure)
    pass_vals = integrate_spot_intensities(img_pass, cam_pts, roi=roi)

    best_score = -np.inf
    best_xy = np.array([x0, y0], dtype=np.float32)

    for x in xs:
        for y in ys:
            if x < 0 or x >= DMD_WIDTH or y < 0 or y >= DMD_HEIGHT:
                continue

            mask = block_single_dmd_point_mask(x, y, radius_px=aperture_radius)
            mask = apply_zero_block(mask, zero_dmd_xy, zero_radius_px=zero_radius_px)

            dmd.send_binary_plane(plane, mask)
            dmd.show_plane(plane)

            time.sleep(0.04)
            img = safe_get_image(cam, exposure=exposure)
            vals = integrate_spot_intensities(img, cam_pts, roi=roi)

            drop = pass_vals - vals
            drop = np.clip(drop, 0, None)

            target_drop = drop[spot_index]
            other_drop = np.delete(drop, spot_index)
            leakage = np.max(other_drop) if len(other_drop) else 0

            score = target_drop - leakage_weight * leakage

            if score > best_score:
                best_score = score
                best_xy = np.array([x, y], dtype=np.float32)

    print(
        f"spot {spot_index}: coarse {initial_xy} -> refined {best_xy}, score={best_score:.3g}",
        flush=True,
    )

    return best_xy


def refine_subset_blocking(
    dmd,
    cam,
    cam_pts,
    coarse_dmd_pts,
    indices,
    search_radius=30,
    step=10,
    aperture_radius=15,
    plane=240,
    pass_plane=200,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=60,
):
    refined = []

    for idx in indices:
        xy = refine_dmd_point_for_spot_blocking(
            dmd=dmd,
            cam=cam,
            cam_pts=cam_pts,
            spot_index=idx,
            initial_xy=coarse_dmd_pts[idx],
            search_radius=search_radius,
            step=step,
            aperture_radius=aperture_radius,
            plane=plane,
            pass_plane=pass_plane,
            exposure=exposure,
            roi=roi,
            zero_dmd_xy=zero_dmd_xy,
            zero_radius_px=zero_radius_px,
            leakage_weight=0.5,
        )
        refined.append(xy)

    return np.array(refined, dtype=np.float32)


# ============================================================
# Affine camera -> DMD mapping
# ============================================================

def fit_cam_to_dmd_affine(cam_pts_subset, dmd_pts_subset):
    """
    Fit affine map from camera xy to DMD xy.
    Need at least 3 non-collinear spots.
    """
    cam_pts_subset = np.asarray(cam_pts_subset, dtype=np.float32)
    dmd_pts_subset = np.asarray(dmd_pts_subset, dtype=np.float32)

    if len(cam_pts_subset) < 3:
        raise ValueError("Need at least 3 points for affine fit.")

    M, inliers = cv2.estimateAffine2D(cam_pts_subset, dmd_pts_subset)
    if M is None:
        raise RuntimeError("Affine camera->DMD fit failed.")

    return M, inliers


def apply_affine(M, pts):
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    return pts_h @ M.T


# ============================================================
# Movie: cumulative OFF then ON, with 0th-order blocked
# ============================================================

def make_cumulative_off_then_on_movie_offpass(
    dmd,
    cam,
    dmd_pts,
    cam_pts=None,
    pass_plane=200,
    block_plane=201,
    work_plane=230,
    spot_radius_px=15,
    zero_dmd_xy=None,
    zero_radius_px=60,
    exposure=0.0001,
    settle_s=0.15,
    repeats_per_state=2,
    hold_all_off_frames=4,
    hold_all_on_frames=4,
    out_path="dmdsuite/calibration/circle_offpass_cumulative_movie.gif",
):
    """
    OFF-pass movie:
      1. all spots pass, 0th order blocked
      2. cumulative OFF: add white blocks one-by-one
      3. all spots blocked
      4. cumulative ON: add black pass holes one-by-one
      5. all spots pass again
    """

    dmd_pts = np.asarray(dmd_pts, dtype=np.float32)
    nspots = len(dmd_pts)

    if cam_pts is not None and len(cam_pts) == nspots:
        order = sort_spots_clockwise(cam_pts)
    else:
        order = np.arange(nspots)

    print("Movie sweep order:", order, flush=True)

    frames_raw = []
    labels = []

    def grab(label):
        time.sleep(settle_s)
        for _ in range(repeats_per_state):
            img = safe_get_image(cam, exposure=exposure)
            frames_raw.append(img.astype(np.float32))
            labels.append(label)

    # All pass except zero.
    zero_block_mask = apply_zero_block(
        dmd_pass_all_pattern(),
        zero_dmd_xy,
        zero_radius_px=zero_radius_px,
    )
    dmd.send_binary_plane(pass_plane, zero_block_mask)
    dmd.show_plane(pass_plane)
    grab("ALL ON / zero blocked")

    # Cumulative OFF: start pass, add blocks.
    off_so_far = []
    for k, idx in enumerate(order):
        off_so_far.append(idx)

        mask = block_spots_mask(
            dmd_pts,
            block_indices=off_so_far,
            radius_px=spot_radius_px,
        )
        mask = apply_zero_block(mask, zero_dmd_xy, zero_radius_px=zero_radius_px)

        dmd.send_binary_plane(work_plane, mask)
        dmd.show_plane(work_plane)

        print(f"OFF step {k+1}/{nspots}: blocked {off_so_far}", flush=True)
        grab(f"OFF {k+1}/{nspots}")

    # All blocked.
    dmd.send_binary_plane(block_plane, dmd_block_all_pattern())
    dmd.show_plane(block_plane)
    for _ in range(hold_all_off_frames):
        grab("ALL OFF")

    # Cumulative ON: start block all, add pass holes.
    on_so_far = []
    for k, idx in enumerate(order):
        on_so_far.append(idx)

        mask = pass_selected_spots_mask(
            dmd_pts,
            pass_indices=on_so_far,
            radius_px=spot_radius_px,
        )
        # No need to add zero block here because background is already white/block.
        # But apply anyway to ensure zero stays blocked.
        mask = apply_zero_block(mask, zero_dmd_xy, zero_radius_px=zero_radius_px)

        dmd.send_binary_plane(work_plane, mask)
        dmd.show_plane(work_plane)

        print(f"ON step {k+1}/{nspots}: passed {on_so_far}", flush=True)
        grab(f"ON {k+1}/{nspots}")

    # All pass except zero again.
    dmd.send_binary_plane(pass_plane, zero_block_mask)
    dmd.show_plane(pass_plane)
    for _ in range(hold_all_on_frames):
        grab("ALL ON / zero blocked")

    # Fixed scaling
    stack = np.stack(frames_raw)
    vmin = np.percentile(stack, 0.0)
    vmax = np.percentile(stack, 99.999)
    if vmax <= vmin:
        vmax = vmin + 1

    frames_uint8 = []
    for img, label in zip(frames_raw, labels):
        img8 = (img - vmin) / (vmax - vmin)
        img8 = np.clip(img8, 0, 1)
        img8 = (img8 * 255).astype(np.uint8)

        rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
        cv2.putText(
            rgb,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        frames_uint8.append(rgb)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, frames_uint8, duration=0.22)
    print(f"Saved movie: {out_path}", flush=True)

    return order

def check_zero_order_block(
    dmd,
    cam,
    zero_cam_xy,
    zero_dmd_xy,
    img_zero_before,
    zero_block_plane,
    zero_block_radius_px=60,
    exposure=0.0001,
    roi=12,
    save_path="dmdsuite/calibration/zero_order_block_check.npz",
    plot=True,
):
    """
    Apply the 0th-order DMD block, take an image, and compare before/after
    intensity at the original 0th-order camera position.
    """

    zero_block_mask = apply_zero_block(
        dmd_pass_all_pattern(),
        zero_dmd_xy,
        zero_radius_px=zero_block_radius_px,
    )

    dmd.send_binary_plane(zero_block_plane, zero_block_mask)
    dmd.show_plane(zero_block_plane)

    time.sleep(0.2)
    img_zero_after = safe_get_image(cam, exposure=exposure)

    zero_before_val = integrate_spot_intensities(
        img_zero_before,
        np.array([zero_cam_xy], dtype=np.float32),
        roi=roi,
    )[0]

    zero_after_val = integrate_spot_intensities(
        img_zero_after,
        np.array([zero_cam_xy], dtype=np.float32),
        roi=roi,
    )[0]

    residual_fraction = zero_after_val / zero_before_val if zero_before_val > 0 else np.nan

    print(f"0th-order ROI intensity before block: {zero_before_val:.3g}")
    print(f"0th-order ROI intensity after block:  {zero_after_val:.3g}")
    print(f"Residual fraction after block:        {residual_fraction:.3f}")

    if plot:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img_zero_before, cmap="gray")
        plt.plot(zero_cam_xy[0], zero_cam_xy[1], "rx", markersize=12)
        plt.title("Before zero-order block")

        plt.subplot(1, 2, 2)
        plt.imshow(img_zero_after, cmap="gray")
        plt.plot(zero_cam_xy[0], zero_cam_xy[1], "rx", markersize=12)
        plt.title("After zero-order block")

        plt.tight_layout()
        plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        zero_cam_xy=zero_cam_xy,
        zero_dmd_xy=zero_dmd_xy,
        zero_block_radius_px=zero_block_radius_px,
        image_before=img_zero_before,
        image_after=img_zero_after,
        roi=roi,
        roi_intensity_before=zero_before_val,
        roi_intensity_after=zero_after_val,
        residual_fraction=residual_fraction,
    )

    print(f"Saved zero-order block check to: {save_path}")

    return img_zero_after, zero_before_val, zero_after_val, residual_fraction

def estimate_block_radius_from_scan(
    x_positions,
    x_drop,
    y_positions,
    y_drop,
    spot_index=0,
    safety_factor=1.3,
    min_radius=10,
    max_radius=150,
):
    """
    Estimate DMD blocking radius in mirrors/pixels from x/y drop scans.
    DMD pixel ≈ one micromirror.

    x_drop, y_drop can be 1D or 2D.
    If 2D, shape is [num_positions, num_spots].
    """

    x_positions = np.asarray(x_positions, dtype=np.float32)
    y_positions = np.asarray(y_positions, dtype=np.float32)
    x_drop = np.asarray(x_drop, dtype=np.float32)
    y_drop = np.asarray(y_drop, dtype=np.float32)

    if x_drop.ndim == 2:
        x_vals = x_drop[:, spot_index]
    else:
        x_vals = x_drop

    if y_drop.ndim == 2:
        y_vals = y_drop[:, spot_index]
    else:
        y_vals = y_drop

    def fwhm(pos, vals):
        vals = vals - np.min(vals)
        peak_idx = int(np.argmax(vals))
        peak = vals[peak_idx]

        if peak <= 0:
            return np.nan

        half = 0.5 * peak

        left = peak_idx
        while left > 0 and vals[left] > half:
            left -= 1

        right = peak_idx
        while right < len(vals) - 1 and vals[right] > half:
            right += 1

        return float(pos[right] - pos[left])

    fwhm_x = fwhm(x_positions, x_vals)
    fwhm_y = fwhm(y_positions, y_vals)

    max_fwhm = np.nanmax([fwhm_x, fwhm_y])

    if np.isnan(max_fwhm) or max_fwhm <= 0:
        radius = min_radius
    else:
        radius = int(np.ceil(0.5 * max_fwhm * safety_factor))

    radius = int(np.clip(radius, min_radius, max_radius))

    print(
        f"spot {spot_index}: FWHM x={fwhm_x:.1f}, y={fwhm_y:.1f} mirrors "
        f"=> recommended radius={radius} mirrors"
    )

    return radius

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Rotated 2D Gaussian for local spot fitting.
    Returns flattened array for scipy curve_fit.
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)

    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (2 * sigma_y**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (2 * sigma_y**2)

    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return g.ravel()


def fit_gaussian_2d_local(image, center_xy, size=6):
    """
    Fit a local 2D Gaussian around center_xy.
    center_xy is [x, y].
    Returns optimized [x, y]. If fit fails, returns original center.
    """
    img = np.asarray(image).astype(np.float32)
    h, w = img.shape

    x0, y0 = center_xy
    x0 = float(x0)
    y0 = float(y0)

    x_min = max(0, int(round(x0 - size)))
    x_max = min(w, int(round(x0 + size + 1)))
    y_min = max(0, int(round(y0 - size)))
    y_max = min(h, int(round(y0 + size + 1)))

    local = img[y_min:y_max, x_min:x_max]

    if local.size < 9:
        return np.array([x0, y0], dtype=np.float32), None

    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    xx, yy = np.meshgrid(x, y)

    offset0 = float(np.percentile(local, 20))
    amp0 = float(local.max() - offset0)

    if amp0 <= 0:
        return np.array([x0, y0], dtype=np.float32), None

    initial_guess = (
        amp0,        # amplitude
        x0,          # xo
        y0,          # yo
        2.0,         # sigma_x
        2.0,         # sigma_y
        0.0,         # theta
        offset0,     # offset
    )

    bounds = (
        [0, x_min, y_min, 0.5, 0.5, -np.pi, 0],
        [np.inf, x_max, y_max, 20.0, 20.0, np.pi, np.inf],
    )

    try:
        popt, _ = curve_fit(
            gaussian_2d,
            (xx, yy),
            local.ravel(),
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000,
        )

        _, xo, yo, sigma_x, sigma_y, theta, offset = popt
        refined_xy = np.array([xo, yo], dtype=np.float32)

        return refined_xy, popt

    except Exception:
        return np.array([x0, y0], dtype=np.float32), None


def detect_top_n_spots(
    img,
    n=3,
    threshold_percentile=99.0,
    min_area=3,
    max_area=5000,
    zero_order_xy=None,
    zero_order_exclusion_radius=60,
    min_separation_px=25,
    refine_roi=6,
    plot=True,
    title="Top detected spots",
    blob_sigma=2.0,
    smoothing_sigma=0.5,
    integration_radius=5,
):
    """
    Detect strongest n spots using LoG blob detection, then refine each center
    using local 2D Gaussian fitting.

    Returns Nx2 camera coordinates [x, y].

    If n=None, returns all accepted blobs after filtering and duplicate removal.
    """

    imgf = np.asarray(img).astype(np.float32)

    # Smooth lightly for blob detection only.
    if smoothing_sigma is not None and smoothing_sigma > 0:
        img_smooth = gaussian(imgf, sigma=smoothing_sigma, preserve_range=True)
    else:
        img_smooth = imgf.copy()

    # blob_log threshold is absolute, so convert percentile to absolute threshold.
    threshold_abs = np.percentile(img_smooth, threshold_percentile)

    blobs = blob_log(
        img_smooth,
        min_sigma=blob_sigma,
        max_sigma=blob_sigma,
        num_sigma=1,
        threshold=threshold_abs,
    )

    candidates = []

    for blob in blobs:
        y_blob, x_blob, sigma = blob
        radius = float(sigma) * np.sqrt(2)
        area_est = np.pi * radius**2

        if not (min_area <= area_est <= max_area):
            continue

        # Exclude 0th order if provided.
        if zero_order_xy is not None:
            zx, zy = zero_order_xy
            if np.hypot(x_blob - zx, y_blob - zy) < zero_order_exclusion_radius:
                continue

        # Integrated intensity near blob.
        rr, cc = disk(
            (y_blob, x_blob),
            integration_radius,
            shape=imgf.shape,
        )
        total = float(np.sum(imgf[rr, cc]))

        candidates.append(
            {
                "xy_blob": np.array([x_blob, y_blob], dtype=np.float32),
                "radius": radius,
                "area": area_est,
                "total": total,
            }
        )

    # Sort by brightness.
    candidates = sorted(candidates, key=lambda c: c["total"], reverse=True)

    selected_raw = []
    selected_candidates = []

    for c in candidates:
        xy = c["xy_blob"]

        if selected_raw:
            dists = [np.linalg.norm(xy - s) for s in selected_raw]
            if np.min(dists) < min_separation_px:
                continue

        selected_raw.append(xy)
        selected_candidates.append(c)

        if n is not None and len(selected_raw) == n:
            break

    selected_raw = np.array(selected_raw, dtype=np.float32)

    if n is not None and len(selected_raw) < n:
        print(f"WARNING: requested {n} spots, but only found {len(selected_raw)}.")

    # Refine each selected point by local Gaussian fit.
    refined = []
    fit_params = []

    for xy in selected_raw:
        xy_refined, popt = fit_gaussian_2d_local(
            imgf,
            center_xy=xy,
            size=refine_roi,
        )
        refined.append(xy_refined)
        fit_params.append(popt)

    refined = np.array(refined, dtype=np.float32)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap="gray")

        for k, (raw_xy, ref_xy) in enumerate(zip(selected_raw, refined)):
            # raw blob position
            plt.plot(raw_xy[0], raw_xy[1], "yo", markersize=4)
            # Gaussian-refined position
            plt.plot(ref_xy[0], ref_xy[1], "ro", markersize=3)
            plt.text(ref_xy[0] + 5, ref_xy[1] + 5, f"{k}", color="red")

        if zero_order_xy is not None:
            plt.plot(zero_order_xy[0], zero_order_xy[1], "gx", markersize=6)
            plt.text(
                zero_order_xy[0] + 5,
                zero_order_xy[1] + 5,
                "0th excluded",
                color="lime",
            )

        plt.title(f"{title}: yellow=blob, red=Gaussian fit")
        plt.tight_layout()
        plt.show()

    return refined

def dense_center_scan_for_spot_blocking(
    dmd,
    cam,
    cam_pts,
    spot_index,
    center_guess_xy,
    search_radius=30,
    step=3,
    aperture_radius=15,
    plane=246,
    pass_plane=202,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=30,
    leakage_weight=0.7,
    plot=True,
):
    """
    Dense 2D scan around a coarse DMD center.
    Finds best DMD center for blocking one camera spot.

    Score = target_drop_fraction - leakage_weight * max_other_drop_fraction.
    """

    center_guess_xy = np.asarray(center_guess_xy, dtype=np.float32)
    x0, y0 = center_guess_xy

    xs = np.arange(x0 - search_radius, x0 + search_radius + step, step)
    ys = np.arange(y0 - search_radius, y0 + search_radius + step, step)

    # Reference pass image
    dmd.show_plane(pass_plane)
    time.sleep(0.15)
    img_pass = safe_get_image(cam, exposure=exposure)

    pass_vals = integrate_spot_intensities(img_pass, cam_pts, roi=roi)
    pass_vals_safe = np.maximum(pass_vals, 1.0)

    score_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    target_map = np.zeros_like(score_map)
    leakage_map = np.zeros_like(score_map)

    best_score = -np.inf
    best_xy = center_guess_xy.copy()
    best_target = 0.0
    best_leakage = 0.0

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if x < 0 or x >= DMD_WIDTH or y < 0 or y >= DMD_HEIGHT:
                score_map[iy, ix] = np.nan
                continue

            mask = block_single_dmd_point_mask(
                x,
                y,
                radius_px=aperture_radius,
            )
            mask = apply_zero_block(
                mask,
                zero_dmd_xy,
                zero_radius_px=zero_radius_px,
            )

            dmd.send_binary_plane(plane, mask)
            dmd.show_plane(plane)
            time.sleep(0.035)

            img = safe_get_image(cam, exposure=exposure)
            vals = integrate_spot_intensities(img, cam_pts, roi=roi)

            drops = pass_vals - vals
            drops = np.clip(drops, 0, None)
            frac = drops / pass_vals_safe

            target = frac[spot_index]
            others = np.delete(frac, spot_index)
            leakage = np.max(others) if len(others) else 0.0

            score = target - leakage_weight * leakage

            score_map[iy, ix] = score
            target_map[iy, ix] = target
            leakage_map[iy, ix] = leakage

            if score > best_score:
                best_score = score
                best_xy = np.array([x, y], dtype=np.float32)
                best_target = target
                best_leakage = leakage

        print(
            f"spot {spot_index}: dense row {iy+1}/{len(ys)}, "
            f"best xy={best_xy}, score={best_score:.3f}",
            flush=True,
        )

    if plot:
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(
            target_map,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="auto",
        )
        plt.colorbar(label="target extinction fraction")
        plt.plot(best_xy[0], best_xy[1], "rx", markersize=10)
        plt.title(f"Spot {spot_index}: target")

        plt.subplot(1, 3, 2)
        plt.imshow(
            leakage_map,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="auto",
        )
        plt.colorbar(label="max leakage fraction")
        plt.plot(best_xy[0], best_xy[1], "rx", markersize=10)
        plt.title(f"Spot {spot_index}: leakage")

        plt.subplot(1, 3, 3)
        plt.imshow(
            score_map,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="auto",
        )
        plt.colorbar(label="score")
        plt.plot(best_xy[0], best_xy[1], "rx", markersize=10)
        plt.title(f"Spot {spot_index}: score")

        plt.tight_layout()
        plt.show()

    print(
        f"spot {spot_index}: center {center_guess_xy} -> {best_xy}, "
        f"target extinction={best_target:.3f}, leakage={best_leakage:.3f}, "
        f"score={best_score:.3f}",
        flush=True,
    )

    return {
        "spot_index": int(spot_index),
        "best_xy": best_xy,
        "best_score": float(best_score),
        "best_target_extinction": float(best_target),
        "best_leakage": float(best_leakage),
        "xs": xs,
        "ys": ys,
        "score_map": score_map,
        "target_map": target_map,
        "leakage_map": leakage_map,
    }


def radius_scan_for_spot_blocking(
    dmd,
    cam,
    cam_pts,
    spot_index,
    dmd_xy,
    radii=np.arange(5, 51, 2),
    plane=247,
    pass_plane=202,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=60,
    leakage_weight=0.7,
    plot=True,
):
    """
    At fixed optimized DMD center, scan disk radius.
    This gives how many mirrors should be flipped.
    """

    dmd_xy = np.asarray(dmd_xy, dtype=np.float32)

    dmd.show_plane(pass_plane)
    time.sleep(0.15)
    img_pass = safe_get_image(cam, exposure=exposure)

    pass_vals = integrate_spot_intensities(img_pass, cam_pts, roi=roi)
    pass_vals_safe = np.maximum(pass_vals, 1.0)

    target_ext = []
    leakage_ext = []
    scores = []

    for r in radii:
        mask = block_single_dmd_point_mask(
            dmd_xy[0],
            dmd_xy[1],
            radius_px=int(r),
        )
        mask = apply_zero_block(
            mask,
            zero_dmd_xy,
            zero_radius_px=zero_radius_px,
        )

        dmd.send_binary_plane(plane, mask)
        dmd.show_plane(plane)
        time.sleep(0.06)

        img = safe_get_image(cam, exposure=exposure)
        vals = integrate_spot_intensities(img, cam_pts, roi=roi)

        drops = pass_vals - vals
        drops = np.clip(drops, 0, None)
        frac = drops / pass_vals_safe

        target = frac[spot_index]
        others = np.delete(frac, spot_index)
        leakage = np.max(others) if len(others) else 0.0
        score = target - leakage_weight * leakage

        target_ext.append(target)
        leakage_ext.append(leakage)
        scores.append(score)

        print(
            f"spot {spot_index}: radius={int(r)} mirrors, "
            f"target extinction={target:.3f}, leakage={leakage:.3f}, score={score:.3f}",
            flush=True,
        )

    target_ext = np.asarray(target_ext)
    leakage_ext = np.asarray(leakage_ext)
    scores = np.asarray(scores)

    best_i = int(np.argmax(scores))
    best_radius = int(radii[best_i])
    best_area = float(np.pi * best_radius**2)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(radii, target_ext, "-o", label="target extinction")
        plt.plot(radii, leakage_ext, "-o", label="max leakage")
        plt.plot(radii, scores, "-o", label="score")
        plt.axvline(best_radius, color="r", linestyle="--", label=f"best r={best_radius}")
        plt.xlabel("DMD disk radius [mirrors/pixels]")
        plt.ylabel("fraction")
        plt.title(f"Spot {spot_index}: radius optimization")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print(
        f"spot {spot_index}: best radius={best_radius} mirrors, "
        f"diameter={2*best_radius+1} mirrors, "
        f"area≈{best_area:.0f} mirrors, "
        f"target extinction={target_ext[best_i]:.3f}, "
        f"leakage={leakage_ext[best_i]:.3f}",
        flush=True,
    )

    return {
        "spot_index": int(spot_index),
        "best_radius": best_radius,
        "best_diameter": int(2 * best_radius + 1),
        "best_area_mirrors": best_area,
        "radii": np.asarray(radii),
        "target_extinction": target_ext,
        "leakage": leakage_ext,
        "scores": scores,
    }


def refine_triangle_centers_and_radii_blocking(
    dmd,
    cam,
    tri_cam_pts,
    tri_coarse_dmd_pts,
    center_search_radius=25,
    center_step=3,
    center_aperture_radius=15,
    radius_values=np.arange(5, 41, 2),
    center_plane=246,
    radius_plane=247,
    pass_plane=202,
    exposure=0.0001,
    roi=8,
    zero_dmd_xy=None,
    zero_radius_px=60,
    leakage_weight=0.7,
):
    """
    For each triangle spot:
      1. dense 2D center refinement around coarse peak
      2. radius/mirror-count scan at optimized center

    Returns:
      refined_dmd_pts
      recommended_movie_radius
      center_reports
      radius_reports
    """

    refined_pts = []
    center_reports = []
    radius_reports = []

    for i in range(len(tri_cam_pts)):
        print("\n" + "=" * 80)
        print(f"Dense center refinement for triangle spot {i}")
        print("=" * 80)

        center_report = dense_center_scan_for_spot_blocking(
            dmd=dmd,
            cam=cam,
            cam_pts=tri_cam_pts,
            spot_index=i,
            center_guess_xy=tri_coarse_dmd_pts[i],
            search_radius=center_search_radius,
            step=center_step,
            aperture_radius=center_aperture_radius,
            plane=center_plane,
            pass_plane=pass_plane,
            exposure=exposure,
            roi=roi,
            zero_dmd_xy=zero_dmd_xy,
            zero_radius_px=zero_radius_px,
            leakage_weight=leakage_weight,
            plot=True,
        )

        refined_xy = center_report["best_xy"]
        refined_pts.append(refined_xy)
        center_reports.append(center_report)

        print("\n" + "-" * 80)
        print(f"Radius/mirror-count optimization for triangle spot {i}")
        print("-" * 80)

        radius_report = radius_scan_for_spot_blocking(
            dmd=dmd,
            cam=cam,
            cam_pts=tri_cam_pts,
            spot_index=i,
            dmd_xy=refined_xy,
            radii=radius_values,
            plane=radius_plane,
            pass_plane=pass_plane,
            exposure=exposure,
            roi=roi,
            zero_dmd_xy=zero_dmd_xy,
            zero_radius_px=zero_radius_px,
            leakage_weight=leakage_weight,
            plot=True,
        )

        radius_reports.append(radius_report)

    refined_pts = np.asarray(refined_pts, dtype=np.float32)
    best_radii = np.array([r["best_radius"] for r in radius_reports], dtype=float)

    recommended_movie_radius = int(np.round(np.median(best_radii)))

    print("\n" + "=" * 80)
    print("Final calibration radius recommendation")
    print("=" * 80)
    print("Triangle best radii:", best_radii)
    print(f"Recommended MOVIE_RADIUS = {recommended_movie_radius} mirrors")
    print(f"Approx mirrors flipped per spot ≈ {np.pi * recommended_movie_radius**2:.0f}")
    print("=" * 80)

    return refined_pts, recommended_movie_radius, center_reports, radius_reports

def detect_triangle_spots_near_expected(
    img,
    expected_pts,
    search_radius=80,
    refine_roi=10,
    bg_percentile=50,
    plot=True,
    title="Triangle spots near expected positions",
):
    """
    Detect triangle spots by searching near expected camera coordinates.

    expected_pts: array of shape (3, 2), each row [x, y].
                  Use the same camera coordinates used to generate the SLM triangle.

    Returns:
        refined_pts: (3, 2) array of refined camera coordinates [x, y]
    """
    imgf = np.asarray(img).astype(np.float32)
    h, w = imgf.shape

    expected_pts = np.asarray(expected_pts, dtype=np.float32)
    refined_pts = []

    bg = np.percentile(imgf, bg_percentile)

    for k, (x_exp, y_exp) in enumerate(expected_pts):
        x_exp_i = int(round(x_exp))
        y_exp_i = int(round(y_exp))

        # Search window around expected point
        x0 = max(0, x_exp_i - search_radius)
        x1 = min(w, x_exp_i + search_radius + 1)
        y0 = max(0, y_exp_i - search_radius)
        y1 = min(h, y_exp_i + search_radius + 1)

        patch = imgf[y0:y1, x0:x1]

        if patch.size == 0:
            raise RuntimeError(f"Empty search patch for triangle spot {k}")

        # Find local brightest pixel inside expected region
        local_y_peak, local_x_peak = np.unravel_index(np.argmax(patch), patch.shape)

        x_peak = x0 + local_x_peak
        y_peak = y0 + local_y_peak

        # Refine around peak with intensity-weighted centroid
        xr0 = max(0, x_peak - refine_roi)
        xr1 = min(w, x_peak + refine_roi + 1)
        yr0 = max(0, y_peak - refine_roi)
        yr1 = min(h, y_peak + refine_roi + 1)

        roi = imgf[yr0:yr1, xr0:xr1]
        weights = roi - bg
        weights = np.clip(weights, 0, None)

        if weights.sum() <= 0:
            x_ref, y_ref = x_peak, y_peak
        else:
            yy, xx = np.mgrid[yr0:yr1, xr0:xr1]
            x_ref = np.sum(xx * weights) / np.sum(weights)
            y_ref = np.sum(yy * weights) / np.sum(weights)

        refined_pts.append([x_ref, y_ref])

        print(
            f"triangle {k}: expected=({x_exp:.1f}, {y_exp:.1f}), "
            f"peak=({x_peak:.1f}, {y_peak:.1f}), "
            f"refined=({x_ref:.2f}, {y_ref:.2f})",
            flush=True,
        )

    refined_pts = np.array(refined_pts, dtype=np.float32)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(img, cmap="gray")

        for k, (p_exp, p_ref) in enumerate(zip(expected_pts, refined_pts)):
            # expected point
            plt.plot(p_exp[0], p_exp[1], "gx", markersize=10)
            # refined detected point
            plt.plot(p_ref[0], p_ref[1], "ro", markersize=6)
            plt.text(p_ref[0] + 5, p_ref[1] + 5, f"{k}", color="red")

            # search box
            rect = plt.Rectangle(
                (p_exp[0] - search_radius, p_exp[1] - search_radius),
                2 * search_radius,
                2 * search_radius,
                fill=False,
                edgecolor="yellow",
                linewidth=1,
            )
            plt.gca().add_patch(rect)

        plt.title(title + " | green=expected, red=refined")
        plt.tight_layout()
        plt.show()

    return refined_pts

def load_triangle_affine_calibration(
    path="dmdsuite/calibration/triangle_affine_offpass.npz",
):
    """
    Load saved triangle calibration.
    This gives camera -> DMD affine mapping for any new SLM pattern.
    """
    data = np.load(path, allow_pickle=True)

    M_cam_to_dmd = data["M_cam_to_dmd"]
    zero_dmd_xy = data["zero_dmd_xy"]

    # Optional fields
    zero_cam_xy = data["zero_cam_xy"] if "zero_cam_xy" in data.files else None

    if "optimized_movie_radius" in data.files:
        movie_radius = int(data["optimized_movie_radius"])
    else:
        movie_radius = 15

    print("Loaded triangle affine calibration:")
    print("M_cam_to_dmd =")
    print(M_cam_to_dmd)
    print("zero_dmd_xy =", zero_dmd_xy)
    print("movie_radius =", movie_radius)

    return M_cam_to_dmd, zero_dmd_xy, zero_cam_xy, movie_radius


def map_new_pattern_camera_points_to_dmd(
    cam_pts,
    M_cam_to_dmd,
):
    """
    Convert new pattern camera coordinates to DMD coordinates.
    """
    cam_pts = np.asarray(cam_pts, dtype=np.float32)
    dmd_pts = apply_affine(M_cam_to_dmd, cam_pts).astype(np.float32)
    return dmd_pts


# ============================================================
# Movie for NEW SLM pattern using saved triangle affine
# ============================================================

if __name__ == "__main__":
    from slmsuite.hardware.cameras.thorlabs import ThorCam
    from utils import kplotlib as kpl
    kpl.init_kplotlib()

    lib_path = r"dmdsuite\Windows_x86_64\DLL_x64\x64\Release\DLP6500_DLL.dll"

    # Planes
    PASS_PLANE = 200          # black/pass
    BLOCK_PLANE = 201         # white/block
    ZERO_BLOCK_PLANE = 202
    MOVIE_PLANE = 230

    # Files
    triangle_calib_path = "dmdsuite/calibration/triangle_affine_offpass.npz"

    # Imaging/movie parameters
    EXPOSURE_PATTERN = 0.0001
    EXPOSURE_MOVIE = 0.0001
    ZERO_BLOCK_RADIUS = 30
    DEFAULT_MOVIE_RADIUS = 25

    dmd = Dmd6500(lib_path)
    cam = ThorCam(serial="26438", verbose=True)

    try:
        dmd.connect(0)

        # Load saved triangle calibration
        M_cam_to_dmd, zero_dmd_xy, zero_cam_xy, saved_movie_radius = (
            load_triangle_affine_calibration(triangle_calib_path)
        )

        MOVIE_RADIUS = saved_movie_radius if saved_movie_radius is not None else DEFAULT_MOVIE_RADIUS
        print("Using MOVIE_RADIUS =", MOVIE_RADIUS)

        # Upload basic planes
        dmd.send_binary_plane(PASS_PLANE, dmd_pass_all_pattern())
        dmd.send_binary_plane(BLOCK_PLANE, dmd_block_all_pattern())

        # Upload zero-order block plane
        zero_block_mask = apply_zero_block(
            dmd_pass_all_pattern(),
            zero_dmd_xy,
            zero_radius_px=ZERO_BLOCK_RADIUS,
        )
        dmd.send_binary_plane(ZERO_BLOCK_PLANE, zero_block_mask)
        dmd.show_plane(ZERO_BLOCK_PLANE)

        input(
            "\nWrite the NEW SLM pattern now and keep the SLM script paused/running.\n"
            "When the pattern is visible with 0th order blocked, press Enter here..."
        )

        # Take camera image of new pattern
        dmd.show_plane(ZERO_BLOCK_PLANE)
        img_pattern = safe_get_image(cam, exposure=EXPOSURE_PATTERN)

        # Detect new pattern spots
        new_cam_pts = detect_spot_centroids(
            img_pattern,
            threshold_percentile=99.6,
            min_area=60,
            max_area=600,
            zero_order_xy=zero_cam_xy,
            zero_order_exclusion_radius=30,
            plot=True,
            title="New pattern spots",
        )

        print("Detected new pattern camera points:")
        print(new_cam_pts)

        # Map camera -> DMD using saved triangle affine
        new_dmd_pts = map_new_pattern_camera_points_to_dmd(
            new_cam_pts,
            M_cam_to_dmd,
        )

        print("New pattern camera -> DMD points:")
        for i, (c, d) in enumerate(zip(new_cam_pts, new_dmd_pts)):
            print(f"spot {i}: camera {c} -> DMD {d}")

        os.makedirs("dmdsuite/calibration", exist_ok=True)
        np.savez(
            "dmdsuite/calibration/new_pattern_from_triangle_affine.npz",
            image_pattern=img_pattern,
            pattern_camera_points=new_cam_pts,
            pattern_dmd_points=new_dmd_pts,
            M_cam_to_dmd=M_cam_to_dmd,
            zero_dmd_xy=zero_dmd_xy,
            zero_cam_xy=zero_cam_xy,
            movie_radius=MOVIE_RADIUS,
        )

        input("\nPattern mapped. Press Enter to make movie...")

        order = make_cumulative_off_then_on_movie_offpass(
            dmd=dmd,
            cam=cam,
            dmd_pts=new_dmd_pts,
            cam_pts=new_cam_pts,
            pass_plane=PASS_PLANE,
            block_plane=BLOCK_PLANE,
            work_plane=MOVIE_PLANE,
            spot_radius_px=MOVIE_RADIUS,
            zero_dmd_xy=zero_dmd_xy,
            zero_radius_px=ZERO_BLOCK_RADIUS,
            exposure=EXPOSURE_MOVIE,
            settle_s=0.15,
            repeats_per_state=2,
            hold_all_off_frames=4,
            hold_all_on_frames=4,
            out_path="dmdsuite/calibration/new_pattern_triangle_affine_movie.gif",
        )

        print("Movie order:", order)

        input("\nDone. Press Enter to block all and exit...")
        dmd.show_plane(BLOCK_PLANE)

    finally:
        dmd.disconnect()
        cam.close()
sys.exit()
# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    from slmsuite.hardware.cameras.thorlabs import ThorCam

    lib_path = r"C:\Users\jkdol\OneDrive\Documents\Github\dioptric\dmdsuite\Windows_x86_64\DLL_x64\x64\Release\DLP6500_DLL.dll"

    # Planes
    PASS_PLANE = 200          # black/pass-all except zero after update
    BLOCK_PLANE = 201         # white/block-all
    ZERO_BLOCK_PLANE = 202    # black/pass + white zero block
    X_SCAN_PLANE = 220
    Y_SCAN_PLANE = 221
    REFINE_PLANE = 240
    MOVIE_PLANE = 230

    # Exposures
    EXPOSURE_ZERO = 0.0001
    EXPOSURE_TRIANGLE = 0.0002
    EXPOSURE_CIRCLE = 0.0001
    EXPOSURE_SCAN = 0.0001
    EXPOSURE_MOVIE = 0.0001

    # Scan ranges
    X_SCAN_POSITIONS = np.arange(875, 905, 2)
    Y_SCAN_POSITIONS = np.arange(445, 475, 2)
    STRIPE_WIDTH = 20

    # Refinement only for triangle spots
    REFINEMENT_SEARCH_RADIUS = 30
    REFINEMENT_STEP = 10
    REFINEMENT_APERTURE_RADIUS = 15

    # Movie
    MOVIE_RADIUS = 25
    ZERO_BLOCK_RADIUS = 40

    dmd = Dmd6500(lib_path)
    cam = ThorCam(serial="26438", verbose=True)

    try:
        dmd.connect(0)

        os.makedirs("dmdsuite/calibration", exist_ok=True)

        # ----------------------------------------------------
        # Step 0: upload basic planes
        # ----------------------------------------------------
        print("Uploading PASS and BLOCK planes for OFF-pass alignment...")
        dmd.send_binary_plane(PASS_PLANE, dmd_pass_all_pattern())
        dmd.send_binary_plane(BLOCK_PLANE, dmd_block_all_pattern())

        # ----------------------------------------------------
        # Step 1: no SLM pattern. Find 0th order.
        # ----------------------------------------------------
        dmd.show_plane(PASS_PLANE)

        input(
            "\nSTEP 1: Make sure NO SLM hologram/pattern is written.\n"
            "DMD is black/pass-all. Confirm only 0th order is visible, then press Enter..."
        )

        img_zero = safe_get_image(cam, exposure=EXPOSURE_ZERO)
        zero_cam_xy = brightest_spot_centroid(img_zero, plot=True)

        print("Finding DMD coordinate of 0th order by blocking-drop scan...")

        zero_dmd_pts, zxpos, zypos, zxdrop, zydrop = find_dmd_xy_for_camera_spots_blocking(
            dmd=dmd,
            cam=cam,
            cam_pts=np.array([zero_cam_xy], dtype=np.float32),
            x_positions=X_SCAN_POSITIONS,
            y_positions=Y_SCAN_POSITIONS,
            stripe_width=STRIPE_WIDTH,
            x_plane=X_SCAN_PLANE,
            y_plane=Y_SCAN_PLANE,
            pass_plane=PASS_PLANE,
            exposure=EXPOSURE_SCAN,
            roi=10,
            zero_dmd_xy=None,
            zero_radius_px=ZERO_BLOCK_RADIUS,
            plot=True,
        )

        zero_dmd_xy = zero_dmd_pts[0]

        ZERO_BLOCK_RADIUS = estimate_block_radius_from_scan(
            x_positions=zxpos,
            x_drop=zxdrop,
            y_positions=zypos,
            y_drop=zydrop,
            spot_index=0,
            safety_factor=1.6,
            min_radius=40,
            max_radius=150,
        )

        print("Using ZERO_BLOCK_RADIUS =", ZERO_BLOCK_RADIUS)

        print("0th order camera xy:", zero_cam_xy)
        print("0th order DMD xy:", zero_dmd_xy)

        # Upload permanent zero block plane.
        zero_block_mask = apply_zero_block(
            dmd_pass_all_pattern(),
            zero_dmd_xy,
            zero_radius_px=ZERO_BLOCK_RADIUS,
        )

        dmd.send_binary_plane(ZERO_BLOCK_PLANE, zero_block_mask)
        dmd.show_plane(ZERO_BLOCK_PLANE)
        time.sleep(0.2)

        input("\nCheck the before/after image. Press Enter if 0th order is sufficiently blocked...")
        # input("\n0th order should now be blocked. Check camera, then press Enter...")

        img_zero_blocked, zero_before_val, zero_after_val, zero_residual = check_zero_order_block(
            dmd=dmd,
            cam=cam,
            zero_cam_xy=zero_cam_xy,
            zero_dmd_xy=zero_dmd_xy,
            img_zero_before=img_zero,
            zero_block_plane=ZERO_BLOCK_PLANE,
            zero_block_radius_px=ZERO_BLOCK_RADIUS,
            exposure=EXPOSURE_ZERO,
            roi=12,
            save_path="dmdsuite/calibration/zero_order_block_check.npz",
            plot=True,
        )

        np.savez(
            "dmdsuite/calibration/zero_order_offpass.npz",
            zero_cam_xy=zero_cam_xy,
            zero_dmd_xy=zero_dmd_xy,
            image_zero=img_zero,
            x_positions=zxpos,
            y_positions=zypos,
            x_drop=zxdrop,
            y_drop=zydrop,
        )

        # ----------------------------------------------------
        # Step 2: write SLM triangle and calibrate affine
        # ----------------------------------------------------
        input(
            "\nSTEP 2: Now write the SLM TRIANGLE pattern in the SLM script.\n"
            "Keep the SLM script paused/running.\n"
            "When triangle spots are visible with 0th order blocked, press Enter here..."
        )

        dmd.show_plane(ZERO_BLOCK_PLANE)
        img_triangle = safe_get_image(cam, exposure=EXPOSURE_TRIANGLE)

        tri_cam_pts = detect_top_n_spots(
            img_triangle,
            n=3,
            threshold_percentile=99.0,
            min_area=3,
            max_area=50000,
            zero_order_xy=zero_cam_xy,
            zero_order_exclusion_radius=40,
            min_separation_px=30,
            refine_roi=6,
            blob_sigma=2.0,
            smoothing_sigma=0.5,
            integration_radius=5,
            plot=True,
            title="Triangle calibration spots",
        )

        if len(tri_cam_pts) < 3:
            raise RuntimeError(
                f"Need at least 3 triangle spots for affine calibration; detected {len(tri_cam_pts)}."
            )

        if len(tri_cam_pts) > 3:
            print(
                f"WARNING: Detected {len(tri_cam_pts)} triangle-like spots. "
                "Using the 3 brightest/first detected spots. If wrong, adjust threshold/exposure."
            )
            tri_cam_pts = tri_cam_pts[:3]

        input("Check triangle labels. Press Enter to scan DMD for triangle spots...")

        tri_coarse_dmd_pts, txpos, typos, txdrop, tydrop = find_dmd_xy_for_camera_spots_blocking(
            dmd=dmd,
            cam=cam,
            cam_pts=tri_cam_pts,
            x_positions= np.arange(200, 1200, 5),
            y_positions= np.arange(200, 800, 5),
            stripe_width=STRIPE_WIDTH,
            x_plane=X_SCAN_PLANE,
            y_plane=Y_SCAN_PLANE,
            pass_plane=ZERO_BLOCK_PLANE,
            exposure=EXPOSURE_SCAN,
            roi=8,
            zero_dmd_xy=zero_dmd_xy,
            zero_radius_px=ZERO_BLOCK_RADIUS,
            plot=True,
        )

        print("Coarse triangle DMD points:")
        for i, (c, d) in enumerate(zip(tri_cam_pts, tri_coarse_dmd_pts)):
            print(f"triangle {i}: camera {c} -> coarse DMD {d}")

        input("Press Enter to do dense center scan + radius optimization for triangle spots...")

        tri_refined_dmd_pts, MOVIE_RADIUS, center_reports, radius_reports = (
            refine_triangle_centers_and_radii_blocking(
                dmd=dmd,
                cam=cam,
                tri_cam_pts=tri_cam_pts,
                tri_coarse_dmd_pts=tri_coarse_dmd_pts,
                center_search_radius=25,
                center_step=3,
                center_aperture_radius=15,
                radius_values=np.arange(5, 41, 2),
                center_plane=246,
                radius_plane=247,
                pass_plane=ZERO_BLOCK_PLANE,
                exposure=EXPOSURE_SCAN,
                roi=8,
                zero_dmd_xy=zero_dmd_xy,
                zero_radius_px=ZERO_BLOCK_RADIUS,
                leakage_weight=0.7,
            )
        )

        print("Refined triangle DMD points and optimized radius:")
        for i, (c, d) in enumerate(zip(tri_cam_pts, tri_refined_dmd_pts)):
            print(f"triangle {i}: camera {c} -> refined DMD {d}")

        print("Updated MOVIE_RADIUS =", MOVIE_RADIUS)

        print("Refined triangle DMD points:")
        for i, (c, d) in enumerate(zip(tri_cam_pts, tri_refined_dmd_pts)):
            print(f"triangle {i}: camera {c} -> refined DMD {d}")

        M_cam_to_dmd, inliers = fit_cam_to_dmd_affine(
            tri_cam_pts,
            tri_refined_dmd_pts,
        )

        print("Affine camera -> DMD matrix:")
        print(M_cam_to_dmd)
        print("Affine inliers:", inliers.ravel() if inliers is not None else None)

        optimized_movie_radius=MOVIE_RADIUS,
        center_best_xy=np.array([r["best_xy"] for r in center_reports]),
        center_best_score=np.array([r["best_score"] for r in center_reports]),
        center_target_extinction=np.array([r["best_target_extinction"] for r in center_reports]),
        center_leakage=np.array([r["best_leakage"] for r in center_reports]),
        radius_best=np.array([r["best_radius"] for r in radius_reports]),
        radius_best_area_mirrors=np.array([r["best_area_mirrors"] for r in radius_reports]),
        np.savez(
            "dmdsuite/calibration/triangle_affine_offpass.npz",
            zero_cam_xy=zero_cam_xy,
            zero_dmd_xy=zero_dmd_xy,
            triangle_camera_points=tri_cam_pts,
            triangle_coarse_dmd_points=tri_coarse_dmd_pts,
            triangle_refined_dmd_points=tri_refined_dmd_pts,
            M_cam_to_dmd=M_cam_to_dmd,
            image_triangle=img_triangle,
            x_positions=txpos,
            y_positions=typos,
            x_drop=txdrop,
            y_drop=tydrop,
            optimized_movie_radius=MOVIE_RADIUS,
            center_best_xy=np.array([r["best_xy"] for r in center_reports]),
            center_best_score=np.array([r["best_score"] for r in center_reports]),
            center_target_extinction=np.array([r["best_target_extinction"] for r in center_reports]),
            center_leakage=np.array([r["best_leakage"] for r in center_reports]),
            radius_best=np.array([r["best_radius"] for r in radius_reports]),
            radius_best_area_mirrors=np.array([r["best_area_mirrors"] for r in radius_reports]),
        )

        input("\nDone. Press Enter to block all and exit...")
        dmd.show_plane(BLOCK_PLANE)

    finally:
        dmd.disconnect()
        cam.close()
