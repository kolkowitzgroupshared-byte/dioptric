"""
subclass for Thorlab SLM hardware control in :mod:`slmsuite`.
Outlines which SLM superclass functions must be implemented.

@author Saroj B Chand

"""

import ctypes
import os
import sys
import time
import warnings

import numpy as np

# sys.path.append('c:/Users/Saroj Chand/Documents/dioptric')
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_CGHDisplay.Thorlabs_EXULUS_CGHDisplay import *
from slmsuite.hardware.Thorlabs_EXULUS_PythonSDK.Thorlabs_EXULUS_Python_SDK.EXULUS_COMMAND_LIB import *

# DEFAULT_SDK_PATH = "C:/Users/Saroj Chand/Documents/dioptric/Thorlabs_EXULUS_PythonSDK"
DEFAULT_SDK_PATH = "C:/Users/matth/GitHub/dioptric/slmsuite/hardware/Thorlabs_EXULUS_PythonSDK/Thorlabs_EXULUS_Python_SDK"

class ThorSLM(SLM):
    """
    Robust Thorlabs EXULUS SLM wrapper for slmsuite.

    Important:
        - __init__ raises errors instead of returning -1.
        - close() is safe to call multiple times.
        - _write_hw() only writes the phase; it does not ask for input.
        - Device/window cleanup happens if initialization partially fails.
    """

    def __init__(
        self,
        serialNumber="00429430",
        screen=2,
        width=1920,
        height=1080,
        baudrate=38400,
        timeout=3,
        max_retries=3,
        retry_delay_s=1.0,
        write_delay_s=2.0,
        verbose=True,
    ):
        self.serialNumber = serialNumber
        self.screen = screen
        self.width = int(width)
        self.height = int(height)
        self.write_delay_s = float(write_delay_s)
        self.verbose = verbose
        
        self.device_hdl = None
        self.window_hdl = None
        self._closed = False

        try:
            # --------------------------------------------------
            # Connect EXULUS device with retries
            # --------------------------------------------------
            last_error = None

            for attempt in range(1, max_retries + 1):
                if self.verbose:
                    print(
                        f"Connecting to Thorlabs SLM {serialNumber} "
                        f"(attempt {attempt}/{max_retries})...",
                        flush=True,
                    )

                hdl = EXULUSOpen(serialNumber, baudrate, timeout)

                if hdl >= 0:
                    self.device_hdl = hdl
                    break

                last_error = hdl

                if self.verbose:
                    print(
                        f"EXULUSOpen failed with code {hdl}. "
                        f"Retrying in {retry_delay_s} s...",
                        flush=True,
                    )

                time.sleep(retry_delay_s)

            if self.device_hdl is None or self.device_hdl < 0:
                raise RuntimeError(
                    f"Failed to connect to Thorlabs SLM serial={serialNumber}. "
                    f"Last error code={last_error}"
                )

            if self.verbose:
                print(f"Connected to Thorlabs SLM {serialNumber}", flush=True)

            # --------------------------------------------------
            # Verify open state
            # --------------------------------------------------
            result = EXULUSIsOpen(serialNumber)
            if result < 0:
                raise RuntimeError(
                    f"EXULUSIsOpen failed for serial={serialNumber}, code={result}"
                )

            if self.verbose:
                print("EXULUS is open.", flush=True)

            # --------------------------------------------------
            # Communication check
            # --------------------------------------------------
            code = [0]
            code_list = {
                6: "Acknowledge",
                9: "Not Acknowledge",
                187: "SPI_Busy",
            }

            result = EXULUSCheckCommunication(self.device_hdl, code)
            if result < 0:
                raise RuntimeError(
                    f"EXULUSCheckCommunication failed with code={result}"
                )

            if self.verbose:
                print(
                    "Communication:",
                    code_list.get(code[0], f"Unknown code {code[0]}"),
                    flush=True,
                )

            if code[0] == 187:
                # SPI busy can happen right after reconnect.
                # Wait and check once more.
                time.sleep(1.0)
                code = [0]
                result = EXULUSCheckCommunication(self.device_hdl, code)

                if result < 0 or code[0] == 187:
                    raise RuntimeError(
                        f"SLM communication still busy after retry. "
                        f"result={result}, code={code[0]}"
                    )

            # --------------------------------------------------
            # Initialize slmsuite superclass
            # --------------------------------------------------
            super().__init__(
                self.width,
                self.height,
                bitdepth=8,
                dx_um=8,
                dy_um=8,
            )

            # --------------------------------------------------
            # Create CGH display window
            # --------------------------------------------------
            if self.verbose:
                print("Creating CGH display window...", flush=True)

            self.window_hdl = CghDisplayCreateWindow(
                self.screen,
                self.width,
                self.height,
                "SLM window",
            )

            if self.window_hdl < 0:
                raise RuntimeError(
                    f"CghDisplayCreateWindow failed with code={self.window_hdl}"
                )

            result = CghDisplaySetWindowInfo(
                self.window_hdl,
                self.width,
                self.height,
                1,
            )

            if result < 0:
                raise RuntimeError(
                    f"CghDisplaySetWindowInfo failed with code={result}"
                )

            # Show blank/zero phase initially.
            blank = np.zeros((self.height, self.width), dtype=np.uint8)
            self._show_uint8(blank)

            if self.verbose:
                print("SLM window created and blank phase shown.", flush=True)

        except Exception:
            # Very important: cleanup partial initialization.
            self.close()
            raise

    def _show_uint8(self, phase_u8):
        """
        Low-level direct display call.
        phase_u8 must be shape (height, width), dtype uint8.
        """
        if self.window_hdl is None or self.window_hdl < 0:
            raise RuntimeError("SLM window is not open.")

        arr = np.asarray(phase_u8)

        if arr.shape != (self.height, self.width):
            raise ValueError(
                f"Expected phase shape {(self.height, self.width)}, got {arr.shape}"
            )

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        arr = np.ascontiguousarray(arr)

        ptr = ctypes.cast(
            arr.ctypes.data,
            ctypes.POINTER(ctypes.c_ubyte),
        )

        result = CghDisplayShowWindow(self.window_hdl, ptr)

        if result < 0:
            raise RuntimeError(f"CghDisplayShowWindow failed with code={result}")

        return result

    def _write_hw(self, phase):
        """
        Low-level hardware interface used by slmsuite.

        The sleep after CghDisplayShowWindow is needed because the Thorlabs/Windows
        display update can be asynchronous; without a short delay, the next code path
        may continue before the SLM panel has fully latched the new frame.
        """
        phase_u8 = np.asarray(phase).astype(np.uint8)

        self._show_uint8(phase_u8)

        if self.write_delay_s > 0:
            time.sleep(self.write_delay_s)

        return 0

    def close(self):
        """
        Close SLM window and device.

        Safe to call multiple times.
        """
        if getattr(self, "_closed", False):
            return

        # Close window first.
        try:
            if getattr(self, "window_hdl", None) is not None and self.window_hdl >= 0:
                if self.verbose:
                    print("Closing SLM display window...", flush=True)
                CghDisplayCloseWindow(self.window_hdl)
        except Exception as exc:
            print(f"Warning: failed to close SLM window: {exc}", flush=True)
        finally:
            self.window_hdl = None

        # Then close EXULUS device.
        try:
            if getattr(self, "device_hdl", None) is not None and self.device_hdl >= 0:
                if self.verbose:
                    print("Closing EXULUS device...", flush=True)
                EXULUSClose(self.device_hdl)
        except Exception as exc:
            print(f"Warning: failed to close EXULUS device: {exc}", flush=True)
        finally:
            self.device_hdl = None

        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def info(verbose=True):
        """
        List detected EXULUS devices.
        """
        devs = EXULUSListDevices()

        if verbose:
            print("Detected EXULUS devices:")
            print(devs)

        return devs