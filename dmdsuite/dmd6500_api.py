# -*- coding: utf-8 -*-
"""
Minimal import-safe Python wrapper for the BBS DLP6500 ALC controller.

Keep this file hardware/API-only.
Do not add camera, plotting, calibration, movie, or test code here.
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------
# DMD geometry
# ---------------------------------------------------------------------

DMD_WIDTH = 1920
DMD_HEIGHT = 1080
PLANE_PADDED_WIDTH = 2048
PLANE_BYTES = PLANE_PADDED_WIDTH * DMD_HEIGHT // 8

SEQ_BUFFER_SIZE = 131072
IDLE_SEQUENCE_START = SEQ_BUFFER_SIZE


# ---------------------------------------------------------------------
# Sequence / command constants
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Errors / triggers
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Main DMD wrapper
# ---------------------------------------------------------------------

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

        self.lib.GetFirmwareVersion.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
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
            raise DmdError(
                f"Requested device index {index}, but not enough controllers found"
            )

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
        self._check(
            self.lib.GetFirmwareVersion(self._require_handle(), ctypes.byref(version)),
            "GetFirmwareVersion",
        )
        return int(version.value)

    @staticmethod
    def _as_gray_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)

        if arr.shape != (DMD_HEIGHT, DMD_WIDTH):
            raise ValueError(
                f"Expected image shape {(DMD_HEIGHT, DMD_WIDTH)}, got {arr.shape}"
            )

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
        """
        Upload one 8-bit grayscale image.

        The vendor API expands the image into 8 consecutive bitplanes,
        starting at start_plane.
        """
        img = self._as_gray_image(image_u8)
        ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        self._check(
            self.lib.SendImageMono(self._require_handle(), int(start_plane), ptr),
            "SendImageMono",
        )

    # def calc_binary_plane(self, image_u8: np.ndarray, bitlevel: int = 0) -> np.ndarray:
    #     gray = self._as_gray_image(image_u8).reshape(-1)

    #     plane = np.zeros(PLANE_BYTES, dtype=np.uint8)

    #     self.lib.CalcPlane(
    #         gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    #         plane.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    #         int(bitlevel),
    #     )

    #     return plane

    # def send_binary_plane(
    #     self,
    #     plane_nr: int,
    #     binary_image: np.ndarray,
    #     bitlevel: int = 0,
    # ) -> None:
    #     plane = self.calc_binary_plane(binary_image, bitlevel=bitlevel)
    #     ptr = plane.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    #     self._check(
    #         self.lib.SendPlane(self._require_handle(), int(plane_nr), ptr),
    #         "SendPlane",
    #     )
    
    def calc_binary_plane_python(
        self,
        image_u8: np.ndarray,
        bitlevel: int = 0,
        bitorder: str = "big",
    ) -> np.ndarray:
        """
        Pack a 1920 x 1080 uint8 binary image into the DMD's padded 2048-bit-wide
        bitplane format, without calling the vendor CalcPlane DLL function.

        This avoids hangs observed inside LabRAD when using CalcPlane.
        """
        img = self._as_gray_image(image_u8)

        # Convert to binary bits.
        # For normal binary masks, any nonzero value is ON.
        if bitlevel is None:
            bits = img > 0
        else:
            bits = ((img >> int(bitlevel)) & 1).astype(bool)

        # DMD rows are padded from 1920 bits to 2048 bits.
        padded = np.zeros((DMD_HEIGHT, PLANE_PADDED_WIDTH), dtype=np.uint8)
        padded[:, :DMD_WIDTH] = bits.astype(np.uint8)

        # Pack 2048 bits per row into 256 bytes per row.
        plane_2d = np.packbits(padded, axis=1, bitorder=bitorder)

        plane = np.ascontiguousarray(plane_2d.reshape(-1).astype(np.uint8))

        if plane.size != PLANE_BYTES:
            raise DmdError(
                f"Packed plane has {plane.size} bytes, expected {PLANE_BYTES}"
            )

        return plane


    def calc_binary_plane(self, image_u8: np.ndarray, bitlevel: int = 0) -> np.ndarray:
        """
        Default to Python packing instead of vendor CalcPlane.
        """
        return self.calc_binary_plane_python(
            image_u8=image_u8,
            bitlevel=bitlevel,
            bitorder="big",
        )


    def send_binary_plane(
        self,
        plane_nr: int,
        binary_image: np.ndarray,
        bitlevel: int = 0,
    ) -> None:
        """
        Upload a binary plane using Python-packed bitplane buffer.
        """
        plane = self.calc_binary_plane(binary_image, bitlevel=bitlevel)
        ptr = plane.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        self._check(
            self.lib.SendPlane(self._require_handle(), int(plane_nr), ptr),
            "SendPlane",
        )

    def show_plane(self, plane_nr: int) -> None:
        handle = self._require_handle()

        # Stop any sequence before static display.
        self.stop_sequence()

        self._check(self.lib.LoadPlaneToDLP(handle, int(plane_nr)), "LoadPlaneToDLP")
        self._check(self.lib.DLP_GlobalMCP(handle), "DLP_GlobalMCP")

    def write_command(self, cmd: int) -> None:
        self._check(
            self.lib.WriteCommand(self._require_handle(), ctypes.c_uint(cmd)),
            "WriteCommand",
        )

    def load_and_clock(self, plane_nr: int) -> None:
        self.write_command(CMD_GLOB_LOAD | LOAD_PLANE_NR | int(plane_nr))
        self.write_command(CMD_GLOB_MIRRORCLOCKING)

    def upload_sequence(
        self,
        commands: Iterable[int],
        startpos: int = 1000,
    ) -> np.ndarray:
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
        self._check(
            self.lib.RunSequence(self._require_handle(), int(startpos)),
            "RunSequence",
        )

    def stop_sequence(self) -> None:
        self._check(self.lib.StopSequence(self._require_handle()), "StopSequence")

    def program_level_gate_sequence(
        self,
        on_plane: int,
        off_plane: int,
        trigger: TriggerPair = TRIG0_LEVEL,
        startpos: int = 1000,
    ) -> np.ndarray:
        """
        Display on_plane while trigger is high, otherwise display off_plane.
        """
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
        """
        Separate charge-prep and readout windows controlled by two trigger lines.
        """
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
        Alternate between plane_a and plane_b with fixed dwell time.
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


# ---------------------------------------------------------------------
# Small safe utility functions
# ---------------------------------------------------------------------

def default_library_path(sdk_root: str | os.PathLike[str]) -> Path:
    """
    Return likely library path from the SDK root.
    """
    sdk_root = Path(sdk_root)
    system = platform.system().lower()

    if system.startswith("win"):
        candidates = [
            sdk_root / "Windows_x86_64" / "DLL_x64" / "x64" / "Release" / "DLP6500_DLL.dll",
            sdk_root / "Windows_x86_64" / "DMD6500_GUI" / "DLP6500_DLL.dll",
        ]

        for p in candidates:
            if p.exists():
                return p

        return candidates[0]

    return sdk_root / "Linux_x86_64" / "Linux_API" / "libbbs_api.so"


def all_off_pattern() -> np.ndarray:
    return np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)


def all_on_pattern() -> np.ndarray:
    return np.full((DMD_HEIGHT, DMD_WIDTH), 255, dtype=np.uint8)


def center_square_pattern(size: int = 500) -> np.ndarray:
    img = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)

    cx = DMD_WIDTH // 2
    cy = DMD_HEIGHT // 2

    x0 = max(0, cx - size // 2)
    x1 = min(DMD_WIDTH, cx + size // 2)
    y0 = max(0, cy - size // 2)
    y1 = min(DMD_HEIGHT, cy + size // 2)

    img[y0:y1, x0:x1] = 255
    return img