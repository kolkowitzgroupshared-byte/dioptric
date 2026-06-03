# -*- coding: utf-8 -*-
"""
Output server for the Teledyne 461 nm laser.

@author: alyssa matthews

### BEGIN NODE INFO
[info]
name = laser_461
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
import time

from labrad.server import LabradServer, setting
from twisted.internet.defer import ensureDeferred

from utils import common


class TopticaDLCProServer(LabradServer):
    name = "laser_461"
    pc_name = socket.gethostname()
    ip_address = "192.168.0.158"
    port = 1998

    def initServer(self):
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Initializing {self.name} on {self.pc_name}...")

        # Establish a persistent socket connection to the laser
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)

        try:
            self.sock.connect((self.ip_address, self.port))
            time.sleep(0.1)

            # Clear the initial greeting/prompt sent by the DLC pro upon connection
            self._read_until_prompt()
            logging.info(
                f"Init complete. Connected to DLC pro at {self.ip_address}:{self.port}"
            )

        except Exception as e:
            logging.error(f"Failed to connect during init: {e}")
            self.sock.close()

    def stopServer(self):
        """
        Called automatically by LabRAD when the server shuts down.
        Safely closes the connection so we don't leave hanging sockets.
        """
        logging.info("Shutting down server, closing socket connection...")
        try:
            self.sock.close()
        except Exception as e:
            logging.debug(e)

    def _read_until_prompt(self):
        """
        Internal helper to read the socket until the Toptica prompt '> ' is found.
        """
        response = b""
        while not response.endswith(b"> "):
            try:
                chunk = self.sock.recv(1024)
                if not chunk:
                    break
                response += chunk
            except socket.timeout:
                logging.warning("Socket read timed out waiting for DLC pro prompt.")
                break
        return response.decode("ascii")

    def _execute_command(self, command):
        """
        Internal helper to send a Scheme instruction and read the response.
        """
        self.sock.sendall((command + "\n").encode("ascii"))
        return self._read_until_prompt()

    # ==========================================
    # LABRAD SETTINGS (Exposed Functions)
    # ==========================================

    @setting(10, "Enable Laser2 Current", enable="b", returns="s")
    def enable_laser2_current(self, c, enable):
        """
        Turns the current control for laser 2 ON or OFF.
        Pass True to enable, False to disable.
        """
        # Toptica control language uses #t for True and #f for False
        scheme_bool = "#t" if enable else "#f"
        command = f"(param-set! 'laser2:dl:cc:enabled {scheme_bool})"

        response = self._execute_command(command)
        return response.strip()

    @setting(11, "Set Laser2 Current", current_mA="v", returns="s")
    def set_laser2_current(self, c, current_mA):
        """
        Sets the laser diode current for laser 2 in mA.
        """
        command = f"(param-set! 'laser2:dl:cc:current-set {current_mA})"

        response = self._execute_command(command)
        return response.strip()


# Instantiate and run the server
__server__ = TopticaDLCProServer()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)
