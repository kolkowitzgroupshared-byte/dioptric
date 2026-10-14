# -*- coding: utf-8 -*-
"""
Output server for the Newport Picomotor controller 8742, driving an XYZ
picomotor stage over USB.

Communication goes through Newport's .NET USB wrapper (UsbDllWrap.dll, installed
with the Newport USB Driver) via pythonnet. The Newport Picomotor Application
must be closed while this server runs since only one process can open the device.

Picomotors are open loop: positions are the controller's step counters, which
reset to 0 at power up. Step size differs between forward and reverse, so
calibrate each direction separately. Axes are numeric (1-4), with x:1, y:2, z:3.

Created on Tuesday Oct 13, 2026

@author: Zhongyuan Liu, with help from Claude Opus 5.5

### BEGIN NODE INFO
[info]
name = picomotor_8742
version = 1.0
description =

[startup]
cmdline = %PYTHON% %FILE%
timeout = 20

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""

import logging
import socket
import sys
import time
import traceback

from labrad.server import LabradServer, setting

from utils import common
from utils import tool_belt as tb

DEFAULT_DLL_DIR = r"C:\Program Files\Newport\Newport USB Driver\Bin"
PRODUCT_ID = 0x4000  # USB product ID of the picomotor controllers
AXES = {"x": 1, "y": 2, "z": 3}
NUM_AXES = 4
POLL_PERIOD = 0.02  # s


class PicomotorController(LabradServer):
    name = "picomotor_8742"
    pc_name = socket.gethostname()

    def initServer(self):
        tb.configure_logging(self)
        self.usb = None
        try:
            config = common.get_config_dict()
            device_ids = config["DeviceIDs"]
            self.serial_num = str(device_ids[f"{self.name}_serial"])
            self.dll_dir = str(device_ids.get(f"{self.name}_dll_dir", DEFAULT_DLL_DIR))
            self._connect()
            logging.info(f"IDN: {self._query('*IDN?')}")
            positions = [self._get_position(ax) for ax in range(1, NUM_AXES + 1)]
            logging.info(f"Axis positions (steps): {positions}")
            logging.info("Init complete")
        except Exception:
            logging.error(traceback.format_exc())
            raise

    def stopServer(self):
        if self.usb is None:
            return
        try:
            for ax in range(1, NUM_AXES + 1):
                self._write(f"{ax}ST")
        except Exception:
            logging.warning(traceback.format_exc())
        try:
            self.usb.CloseDevices()
        except Exception:
            logging.warning(traceback.format_exc())
        self.usb = None

    # region Low-level communication

    def _connect(self):
        import clr

        if self.dll_dir not in sys.path:
            sys.path.append(self.dll_dir)
        clr.AddReference("UsbDllWrap")
        from Newport.USBComm import USB
        from System.Text import StringBuilder

        self.str_bldr = StringBuilder(64)
        self.usb = USB(True)
        if not self.usb.OpenDevices(PRODUCT_ID, True):
            raise RuntimeError(
                f"{self.name}: could not open USB devices. Is the Newport "
                "Picomotor Application still running?"
            )

        device_table = self.usb.GetDeviceTable()
        enumerator = device_table.GetEnumerator()
        keys = []
        while enumerator.MoveNext():
            keys.append(str(enumerator.Key))
        logging.info(f"Discovered device keys: {keys}")

        matches = [key for key in keys if self.serial_num in key]
        if len(matches) == 1:
            self.device_key = matches[0]
        elif len(matches) == 0 and len(keys) == 1:
            self.device_key = keys[0]
            logging.warning(
                f"No device key contains serial {self.serial_num}; "
                f"using the only device found: {self.device_key}"
            )
        else:
            raise RuntimeError(
                f"{self.name}: could not pick a device for serial "
                f"{self.serial_num} from keys {keys}"
            )
        logging.info(f"Using device key {self.device_key}")

    def _write(self, cmd):
        ret = self.usb.Write(self.device_key, cmd)
        if ret != 0:
            raise RuntimeError(f"{self.name}: write '{cmd}' failed with status {ret}")

    def _query(self, cmd):
        self.str_bldr.Remove(0, self.str_bldr.Length)
        ret = self.usb.Query(self.device_key, cmd, self.str_bldr)
        if ret != 0:
            raise RuntimeError(f"{self.name}: query '{cmd}' failed with status {ret}")
        return self.str_bldr.ToString().strip()

    def _check_error(self):
        """Raise if the controller reports an error. TB? returns '<code>, <msg>'"""
        res = self._query("TB?")
        code = res.split(",")[0].strip()
        if code != "0":
            raise RuntimeError(f"{self.name} controller error: {res}")

    def _axis(self, axis):
        axis = int(axis)
        if not 1 <= axis <= NUM_AXES:
            raise ValueError(f"Axis must be 1-{NUM_AXES}, got {axis}")
        return axis

    def _get_position(self, axis):
        return int(self._query(f"{axis}TP?"))

    def _get_velocity(self, axis):
        return int(self._query(f"{axis}VA?"))

    def _is_done(self, axis):
        return self._query(f"{axis}MD?") == "1"

    def _wait_done(self, axis, num_steps):
        """Poll until the axis stops. Stops the axis and raises on timeout"""
        timeout = 5 + 2 * abs(num_steps) / max(self._get_velocity(axis), 1)
        deadline = time.monotonic() + timeout
        while not self._is_done(axis):
            if time.monotonic() > deadline:
                self._write(f"{axis}ST")
                raise RuntimeError(
                    f"{self.name}: axis {axis} did not finish moving within "
                    f"{timeout:.1f} s; motion stopped"
                )
            time.sleep(POLL_PERIOD)

    def _move_rel(self, axis, num_steps, wait=True):
        axis = self._axis(axis)
        num_steps = int(num_steps)
        if num_steps != 0:
            self._write(f"{axis}PR{num_steps}")
            if wait:
                self._wait_done(axis, num_steps)
            self._check_error()
        return self._get_position(axis)

    def _move_abs(self, axis, pos, wait=True):
        axis = self._axis(axis)
        pos = int(round(pos))
        num_steps = pos - self._get_position(axis)
        if num_steps != 0:
            self._write(f"{axis}PA{pos}")
            if wait:
                self._wait_done(axis, num_steps)
            self._check_error()
        return self._get_position(axis)

    # endregion
    # region Step-based settings

    @setting(0, returns="s")
    def get_idn(self, c):
        return self._query("*IDN?")

    @setting(1, axis="i", num_steps="i", wait="b", returns="i")
    def move_rel(self, c, axis, num_steps, wait=True):
        """Move an axis by a relative number of steps. Returns the new position"""
        return self._move_rel(axis, num_steps, wait)

    @setting(2, axis="i", pos="i", wait="b", returns="i")
    def move_abs(self, c, axis, pos, wait=True):
        """Move an axis to an absolute step position. Returns the new position"""
        return self._move_abs(axis, pos, wait)

    @setting(3, axis="i", returns="i")
    def get_position(self, c, axis):
        return self._get_position(self._axis(axis))

    @setting(4, returns="*i")
    def get_xyz(self, c):
        return [self._get_position(AXES[key]) for key in ("x", "y", "z")]

    @setting(5, axis="i", pos="i")
    def set_home(self, c, axis, pos=0):
        """Define the current position of an axis to be pos (default 0)"""
        axis = self._axis(axis)
        self._write(f"{axis}DH{int(pos)}")
        self._check_error()

    @setting(6, axis="i", velocity="i")
    def set_velocity(self, c, axis, velocity):
        """Set the step rate of an axis in steps/s (1-2000)"""
        axis = self._axis(axis)
        if not 1 <= velocity <= 2000:
            raise ValueError(f"Velocity must be 1-2000 steps/s, got {velocity}")
        self._write(f"{axis}VA{velocity}")
        self._check_error()

    @setting(7, axis="i", returns="i")
    def get_velocity(self, c, axis):
        return self._get_velocity(self._axis(axis))

    @setting(8, axis="i", acceleration="i")
    def set_acceleration(self, c, axis, acceleration):
        """Set the acceleration of an axis in steps/s^2"""
        axis = self._axis(axis)
        self._write(f"{axis}AC{acceleration}")
        self._check_error()

    @setting(9, axis="i", returns="b")
    def is_moving(self, c, axis):
        return not self._is_done(self._axis(axis))

    @setting(10, axis="i")
    def stop(self, c, axis=None):
        """Stop an axis with deceleration, or all axes if none is specified"""
        axes = range(1, NUM_AXES + 1) if axis is None else [self._axis(axis)]
        for ax in axes:
            self._write(f"{ax}ST")

    @setting(11, cmd="s")
    def send_command(self, c, cmd):
        """Send a raw command with no response, e.g. '1PR100'"""
        self._write(cmd)

    @setting(12, cmd="s", returns="s")
    def query(self, c, cmd):
        """Send a raw query and return the response, e.g. 'TB?'"""
        return self._query(cmd)

    # endregion
    # region Positioner-compatible settings (units: steps)

    @setting(20, pos="v")
    def write_x(self, c, pos):
        self._move_abs(AXES["x"], pos)

    @setting(21, pos="v")
    def write_y(self, c, pos):
        self._move_abs(AXES["y"], pos)

    @setting(22, pos="v")
    def write_z(self, c, pos):
        self._move_abs(AXES["z"], pos)

    @setting(23, x_pos="v", y_pos="v")
    def write_xy(self, c, x_pos, y_pos):
        self._move_abs(AXES["x"], x_pos)
        self._move_abs(AXES["y"], y_pos)

    @setting(24, x_pos="v", y_pos="v", z_pos="v")
    def write_xyz(self, c, x_pos, y_pos, z_pos):
        self._move_abs(AXES["x"], x_pos)
        self._move_abs(AXES["y"], y_pos)
        self._move_abs(AXES["z"], z_pos)

    @setting(25, returns="*v")
    def read_xy(self, c):
        return [float(self._get_position(AXES[key])) for key in ("x", "y")]

    @setting(26, returns="v")
    def read_z(self, c):
        return float(self._get_position(AXES["z"]))

    @setting(27)
    def reset(self, c):
        for ax in range(1, NUM_AXES + 1):
            self._write(f"{ax}ST")

    # endregion


__server__ = PicomotorController()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
