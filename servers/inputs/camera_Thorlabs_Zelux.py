# -*- coding: utf-8 -*-
"""
Input server for Thorlabs Zelux CMOS camera.

### BEGIN NODE INFO
[info]
name = camera_thorlabs_zelux
version = 1.0
description = Control server for Thorlabs Zelux CMOS camera using TLCameraSDK.
[startup]
cmdline = %PYTHON% %FILE%
timeout = 60
[shutdown]
message = 987654321
timeout = 30
### END NODE INFO
"""

import os
import socket
import time
import logging
import numpy as np

from labrad.server import LabradServer, setting

from utils import common
from utils import tool_belt as tb


def configure_tlcam_dll_path():
    repo = common.get_repo_path()
    dll_path = repo / "slmsuite" / "hardware" / "cameras" / "dlls" / "Native_64_lib"

    dll_path = str(dll_path)
    os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]

    try:
        os.add_dll_directory(dll_path)
    except AttributeError:
        pass

    print("Using Thorlabs DLL path:", dll_path)


configure_tlcam_dll_path()

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, ROI


class CameraThorlabsZelux(LabradServer):
    name = "camera_thorlabs_zelux"
    pc_name = socket.gethostname()

    sdk = None

    def initServer(self):
        tb.configure_logging(self)

        if CameraThorlabsZelux.sdk is None:
            CameraThorlabsZelux.sdk = TLCameraSDK()

        camera_list = CameraThorlabsZelux.sdk.discover_available_cameras()

        if len(camera_list) == 0:
            raise RuntimeError("No Thorlabs cameras found by TLCameraSDK.")

        serial = camera_list[0]
        self.cam = CameraThorlabsZelux.sdk.open_camera(serial)

        self.serial = serial
        self.profile = "single"

        self.width = int(self.cam.image_width_pixels)
        self.height = int(self.cam.image_height_pixels)
        self.bit_depth = int(self.cam.bit_depth)

        self.dtype = np.uint16

        self.cam.exposure_time_us = int(50e3)  # 50 ms default
        self.cam.frames_per_trigger_zero_for_unlimited = 1

        self.is_armed = False

        logging.info(f"Connected to Thorlabs camera serial {serial}")
        print(f"Connected to Thorlabs camera serial {serial}")
        print(f"Image size: {self.width} x {self.height}")
        print(f"Bit depth: {self.bit_depth}")

    def stopServer(self):
        try:
            self.disarm(None)
        except Exception:
            pass

        try:
            self.cam.dispose()
        except Exception:
            pass

        logging.info("Thorlabs camera server stopped.")

    # -------------------------------------------------------------------------
    # Basic info
    # -------------------------------------------------------------------------

    @setting(20, returns="s")
    def get_serial(self, c):
        return str(self.serial)

    @setting(21, returns="*i")
    def get_image_shape(self, c):
        """
        Return image shape as [height, width].
        """
        return [self.height, self.width]

    @setting(22, returns="s")
    def get_dtype(self, c):
        return "uint16"

    @setting(23, returns="i")
    def get_bit_depth(self, c):
        return int(self.bit_depth)

    # -------------------------------------------------------------------------
    # Exposure / ROI
    # -------------------------------------------------------------------------

    @setting(11, exposure_time="v[]")
    def set_exposure_time(self, c, exposure_time):
        was_armed = getattr(self, "is_armed", False)

        if was_armed:
            self.cam.disarm()
            self.is_armed = False

        self.cam.exposure_time_us = int(float(exposure_time) * 1e6)

        if was_armed:
            self.cam.arm(2)
            self.is_armed = True

    @setting(24, returns="v[]")
    def get_exposure_time(self, c):
        """
        Return exposure time in seconds.
        """
        return float(self.cam.exposure_time_us) / 1e6

    @setting(30)
    def clear_roi(self, c):
        """
        Reset to full sensor ROI.
        """
        self.cam.roi = ROI(
            upper_left_x_pixels=0,
            upper_left_y_pixels=0,
            lower_right_x_pixels=self.cam.sensor_width_pixels - 1,
            lower_right_y_pixels=self.cam.sensor_height_pixels - 1,
        )

        self.width = int(self.cam.image_width_pixels)
        self.height = int(self.cam.image_height_pixels)

    @setting(31, x0="i", y0="i", width="i", height="i")
    def set_roi(self, c, x0, y0, width, height):
        """
        Set ROI using x0, y0, width, height.
        """
        x0 = int(x0)
        y0 = int(y0)
        width = int(width)
        height = int(height)

        x1 = x0 + width - 1
        y1 = y0 + height - 1

        self.cam.roi = ROI(
            upper_left_x_pixels=x0,
            upper_left_y_pixels=y0,
            lower_right_x_pixels=x1,
            lower_right_y_pixels=y1,
        )

        self.width = int(self.cam.image_width_pixels)
        self.height = int(self.cam.image_height_pixels)

        print(f"Set ROI: x0={x0}, y0={y0}, width={self.width}, height={self.height}")

    # -------------------------------------------------------------------------
    # Camera control
    # -------------------------------------------------------------------------

    @setting(9)
    def clear_buffer(self, c):
        self._clear_buffer()

    def _clear_buffer(self):
        """
        Drain any pending frames.
        """
        while True:
            frame = self.cam.get_pending_frame_or_null()
            if frame is None:
                break

    @setting(0, num_images="i")
    def arm(self, c, num_images=2):
        """
        Arm camera.

        For software-triggered single image, num_images=2 is usually fine.
        """
        if self.is_armed:
            self.disarm(c)

        self.cam.frames_per_trigger_zero_for_unlimited = 1
        self.cam.arm(max(int(num_images), 2))
        self._clear_buffer()
        self.is_armed = True

    @setting(1)
    def disarm(self, c):
        if getattr(self, "is_armed", False):
            self.cam.disarm()
            self.is_armed = False

    @setting(2, returns="y")
    def read(self, c):
        """
        Read one image and return bytes.

        Client reconstructs with:
            img = np.frombuffer(img_str, dtype=np.uint16).reshape(height, width)
        """
        if not self.is_armed:
            self.arm(c, 2)

        self.cam.issue_software_trigger()

        frame = self.cam.get_pending_frame_or_null()
        t0 = time.time()

        while frame is None:
            time.sleep(0.001)

            if time.time() - t0 > 10:
                raise TimeoutError("Timed out waiting for Thorlabs frame.")

            frame = self.cam.get_pending_frame_or_null()

        img = np.copy(frame.image_buffer).astype(self.dtype)

        self.height, self.width = img.shape

        return img.tobytes()

    @setting(5)
    def reset(self, c):
        self.disarm(c)
        self._clear_buffer()


__server__ = CameraThorlabsZelux()


if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)