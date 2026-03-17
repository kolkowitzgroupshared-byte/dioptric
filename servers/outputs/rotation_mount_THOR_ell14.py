# -*- coding: utf-8 -*-
"""
Output server for the Thorlabs ELL9K filter rotator.

Created on Wed Nov 5 2025

@author: Alyssa Matthews

### BEGIN NODE INFO
[info]
name = rotation_mount_THOR_ell14
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

import serial
from labrad.server import LabradServer, setting
from twisted.internet.defer import inlineCallbacks, returnValue

from utils import common


class RotationMountThorEll14(LabradServer):
    """
    LabRAD Server for the Thorlabs ELL14 Rotation Mount.
    (Simple, robust version with s1 init command)
    """

    name = "rotation_mount_THOR_ell14"
    pc_name = socket.gethostname()

    # -- Communication Parameters --
    port = "COM5"  # CHANGE THIS to your device's COM port
    baudrate = 9600

    # -- ELL14 Specific Parameters --
    # From ELLx manual, page 29: ELL14 Range (counts) = 136533
    COUNTS_PER_DEGREE = 136533 / 360.0

    def initServer(self):
        """
        Initializes the server, connects, and runs the 's1' command.
        """
        logging.info("--- initServer START ---")
        logging.info(
            f"Attempting to connect to ELL14 on {self.port} at {self.baudrate} baud..."
        )
        try:
            # Connect to the serial port
            self.rotator = serial.Serial(
                self.port,
                baudrate=self.baudrate,
                timeout=1.0,  # Set a 1-second read timeout
            )
            logging.info("serial.Serial() call successful.")

            time.sleep(0.1)
            self.rotator.flush()  # Clear any old data
            logging.info("serial.flush() successful.")
            time.sleep(0.1)

            # Find the resonant frequencies of the motor.
            cmd = "0s1".encode()
            logging.info(f"Writing command: {cmd}")
            self.rotator.write(cmd)
            logging.info("Write '0s1' successful.")

            # Wait for device to process and be ready to reply
            time.sleep(0.5)

            # Read the '0GS0' reply
            logging.info("Attempting to read reply...")
            reply = self.rotator.readline().decode().strip()
            logging.info(f"Read reply: '{reply}'")

            if reply != "0GS0":
                logging.warning(
                    f"Unexpected reply to 's1' command. Expected '0GS0', got '{reply}'"
                )

            logging.info("--- initServer SUCCESS: ELL14 connection successful. ---")

        except serial.SerialException as e:
            logging.error(f"--- CRITICAL: SerialException ---")
            logging.error(f"Error opening serial port {self.port}: {e}")
            logging.error(
                "Port may be in use by another program (like Elliptec) or does not exist."
            )
            self.rotator = None
        except Exception as e:
            logging.error(f"--- CRITICAL: Generic Exception ---")
            logging.error(f"Failed during initialization: {e}", exc_info=True)
            self.rotator = None

        if self.rotator is None:
            logging.error("--- initServer FAILED: self.rotator is None ---")

    def stopServer(self):
        """
        Closes the serial connection when the server is stopped.
        """
        if hasattr(self, "rotator") and self.rotator and self.rotator.is_open:
            self.rotator.close()
            logging.info(f"Serial connection on {self.port} closed.")

    @setting(10, "set_angle", angle_deg="v", returns="")
    def set_angle(self, c, angle_deg):
        """
        Sets the rotation mount to a specific angle.

        Args:
            angle_deg (float): The target angle in degrees (0.0 to 360.0).
        """
        if not self.rotator:
            logging.error("Rotator not connected. Cannot set angle.")
            return

        # Clamp angle to valid range
        angle_deg = max(0.0, min(angle_deg, 360.0))

        # Convert angle to counts
        counts = int(round(angle_deg * self.COUNTS_PER_DEGREE))

        # Format as 8-digit hex string
        pos_hex = "{:08X}".format(counts)

        # Create and send command
        cmd_str = "0ma{}".format(pos_hex)  # "Move Absolute"
        cmd_bytes = cmd_str.encode()

        logging.info(f"Sending command: {cmd_str} (to {angle_deg:.2f} deg)")
        try:
            self.rotator.flushInput()
            self.rotator.write(cmd_bytes)
        except Exception as e:
            logging.error(f"Error writing 'ma' command: {e}", exc_info=True)
            return

        logging.info(f"Command sent. Motor is *starting* to move.")

    @setting(11, "get_angle", returns="v")
    def get_angle(self, c):
        """
        Gets the current angle of the rotation mount.

        Returns:
            float: The current angle in degrees, or -1.0 on error.
        """
        if not self.rotator:
            logging.error("Rotator not connected. Cannot get angle.")
            return -1.0

        cmd = "0gp".encode()  # "Get Position"

        try:
            self.rotator.flushInput()
            self.rotator.write(cmd)

            reply_str = self.rotator.readline().decode().strip()

            # Expecting "0PO<pos_hex>" (e.g., "0PO0000A1B2")
            if reply_str.startswith("0PO") and len(reply_str) == 11:
                pos_hex = reply_str[3:]
                counts = int(pos_hex, 16)
                angle_deg = counts / self.COUNTS_PER_DEGREE
                return angle_deg
            else:
                # This catches '' (timeout) or other garbage replies
                logging.error(f"Unexpected reply to 'gp' command: '{reply_str}'")
                return -1.0  # Error value

        except Exception as e:
            logging.error(f"Error reading position: {e}", exc_info=True)
            return -1.0  # Error value


# --- Server Execution ---

__server__ = RotationMountThorEll14()

if __name__ == "__main__":
    from labrad import util

    print("Starting Thorlabs ELL14 Rotation Mount Server (s1 init)...")
    util.runServer(__server__)
