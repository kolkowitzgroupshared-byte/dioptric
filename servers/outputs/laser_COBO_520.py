# -*- coding: utf-8 -*-
"""
Output server for the Cobolt 520 nm laser.

Created on Mon Apr  8 19:50:12 2019

@author: mccambria

Edited 4/20/2026
@author: chemistatcode

### BEGIN NODE INFO
[info]
name = laser_COBO_520
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
import threading
import time

import serial
from labrad.server import setting

from laser_COBO_base import LaserCoboBase
from utils import common


class LaserCobo520(LaserCoboBase):
    wavelength = 520
    name = f"laser_COBO_{wavelength}"

    # ---------------------------------------------------------------- lifecycle

    def initServer(self):
        super().initServer()
        config = common.get_config_dict()
        self._com_port = config["DeviceIDs"][f"{self.name}_com"]
        self._serial = None
        self._serial_lock = threading.Lock()
        self._open_serial()
        logging.debug(f"Serial opened on {self._com_port}")

    def stopServer(self):
        try:
            super().stopServer()
        except Exception:
            pass
        try:
            if self._serial is not None and self._serial.is_open:
                self._serial.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- serial

    def _open_serial(self):
        if self._serial is not None and self._serial.is_open:
            return
        s = serial.Serial(
            self._com_port,
            115200,
            serial.EIGHTBITS,
            serial.PARITY_NONE,
            serial.STOPBITS_ONE,
            timeout=0.5,
        )
        s.reset_input_buffer()
        s.reset_output_buffer()
        self._serial = s

    def _send(self, cmd, expect_response=True):
        """Send `cmd\\r`, drain any stale bytes first, return the response text.

        Holding the COM port open across calls is critical: the previous
        open/close-per-call pattern caused the OS buffer to retain response
        bytes from prior commands, so each `p?` was returning the response to
        an earlier `p X` write. Now we drain any unread bytes before writing
        and read until the response goes quiet.
        """
        with self._serial_lock:
            self._open_serial()
            # Drain stale bytes (errant responses from a prior command).
            n = self._serial.in_waiting
            if n:
                self._serial.read(n)
            self._serial.write(f"{cmd}\r".encode("ascii"))
            if not expect_response:
                return ""
            # readline waits up to `timeout` seconds for `\n`. Cobolt firmware
            # terminates each line with `\r\n`.
            line = self._serial.readline()
            # Drain any trailing bytes (some firmware variants emit echo + OK).
            time.sleep(0.02)
            extra = b""
            if self._serial.in_waiting:
                extra = self._serial.read(self._serial.in_waiting)
            return (line + extra).decode("ascii", errors="replace").strip()

    def _parse_lines(self, resp):
        return [
            ln.strip()
            for ln in resp.replace("\r", "\n").split("\n")
            if ln.strip()
        ]

    def _write_checked(self, cmd):
        """Send a `set` command and verify the laser accepted it.

        Cobolt typically replies "OK" on success; on failure the reply contains
        "Syntax error", "out of range", "not allowed", etc. The previous server
        read one line and discarded it, so rejected commands looked successful.
        """
        resp = self._send(cmd)
        lines = self._parse_lines(resp)
        if not lines:
            raise IOError(f"Cobolt: no response to {cmd!r}")
        last = lines[-1]
        if last.upper() == "OK":
            return
        # Some commands return the new value instead of "OK" — accept numeric.
        try:
            float(last)
            return
        except ValueError:
            pass
        raise IOError(f"Cobolt rejected {cmd!r}: {resp!r}")

    def _query_value(self, cmd):
        """Send a `?` query and return the value line as text."""
        resp = self._send(cmd)
        lines = self._parse_lines(resp)
        if not lines:
            raise IOError(f"Cobolt: no response to {cmd!r}")
        for ln in lines:
            if ln.upper() == "OK":
                continue
            return ln
        raise IOError(f"Cobolt: no value in response to {cmd!r}: {resp!r}")

    # ---------------------------------------------------------------- settings

    @setting(10, power="v[]")
    def set_power(self, c, power):
        """Set the **CW** laser power setpoint in milliwatts (`p <mW>`).

        The Cobolt 520 firmware on this rig uses mW on the wire (verified
        empirically — `slmp 1.0` was accepted as 1 mW, not refused as 1 W).

        Note: in modulation/digital-modulation mode the laser emits at the
        modulation power setpoint, not the CW setpoint. Use
        `set_modulation_power` for sweeps in modulation mode.
        """
        self._write_checked(f"p {float(power):.4f}")

    @setting(11, returns="v[]")
    def get_power(self, c):
        """Read the CW laser power setpoint in milliwatts (`p?`)."""
        return float(self._query_value("p?"))

    @setting(12, returns="v[]")
    def get_actual_power(self, c):
        """Read the actual laser output power in milliwatts (`pa?`).

        In modulation mode this is an instantaneous reading, so it is near 0
        while the modulation TTL gate is LOW and at the modulation-power level
        while the gate is HIGH.
        """
        return float(self._query_value("pa?"))

    @setting(13, power="v[]")
    def set_modulation_power(self, c, power):
        """Set the laser modulation power setpoint in milliwatts (`slmp <mW>`).

        This is the level the laser emits while its modulation TTL gate is
        HIGH. Used in digital/analog modulation mode (the OEM Cobolt GUI sends
        this command when you change the power slider in mod-mode).
        """
        self._write_checked(f"slmp {float(power):.4f}")

    @setting(14, returns="v[]")
    def get_modulation_power(self, c):
        """Read the laser modulation power setpoint in milliwatts (`glmp?`)."""
        return float(self._query_value("glmp?"))

    @setting(15, cmd="s", returns="s")
    def raw_command(self, c, cmd):
        """Send an arbitrary serial command and return the raw response.

        For diagnostics only — exposes the underlying protocol so we can probe
        the laser without redeploying the server.
        """
        return self._send(cmd)


__server__ = LaserCobo520()

if __name__ == '__main__':
    from labrad import util
    util.runServer(__server__)
