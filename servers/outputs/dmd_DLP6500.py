# -*- coding: utf-8 -*-
"""
LabRAD server for DLP6500 DMD spatial gating.

Current optical convention:
    DMD ON  / white / 255 = PASS
    DMD OFF / black / 0   = BLOCK

Recommended startup:
    pass_zero_block

This means:
    white background = pass useful ON-state path
    black disk at zero_dmd_xy = block 0th order

### BEGIN NODE INFO
[info]
name = dmd_DLP6500
version = 1.0
description = DLP6500 DMD spatial gate server

[startup]
cmdline = %PYTHON% %FILE%
timeout = 120

[shutdown]
message = 987654321
timeout = 5
### END NODE INFO
"""
import sys
import time
import json
import logging
import socket
from pathlib import Path
import numpy as np
from labrad.server import LabradServer, setting

from utils import common
from utils import tool_belt as tb

# ---------------------------------------------------------------------
# Import DMD API wrapper
# --------------------------------------------------------------------

from dmdsuite.dmd6500_api import Dmd6500, DMD_WIDTH, DMD_HEIGHT


class DmdDlp6500(LabradServer):
    name = "dmd_DLP6500"
    pc_name = socket.gethostname()

    # Plane numbers
    PASS_PLANE = 200
    BLOCK_PLANE = 201
    ZERO_BLOCK_PLANE = 202
    WORK_PLANE = 230

    # ON-pass convention
    PASS_VALUE = 255   # white / ON = pass
    BLOCK_VALUE = 0    # black / OFF = block

    # -----------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------
    def initServer(self):
        tb.configure_logging(self)

        self.config = common.get_config_dict()
        self.repo_path = common.get_repo_path()
        device_ids = self.config.get("DeviceIDs", {})

        default_dll_path = (
            self.repo_path
            / "dmdsuite"
            / "Windows_x86_64"
            / "DLL_x64"
            / "x64"
            / "Release"
            / "DLP6500_DLL.dll"
        )

        self.dll_path = device_ids.get(
            f"{self.name}_dll",
            str(default_dll_path),
        )

        self.device_index = int(
            device_ids.get(
                f"{self.name}_device_id",
                0,
            )
        )

        # Startup behavior:
        #   "pass"            -> all white/pass
        #   "pass_zero_block" -> all white/pass except zero-order black/block
        #   "block"           -> all black/block
        self.init_state = device_ids.get(
            f"{self.name}_init_state",
            "pass_zero_block",
        )

        # IMPORTANT:
        # This should point to the FINAL chain file, not triangle_affine_onpass.npz.
        # The final chain file contains:
        #   M_cam_to_dmd
        #   zero_dmd_xy
        #   zero_cam_xy
        #   dmd_points / pattern_dmd_points
        self.init_calib_path = device_ids.get(
            f"{self.name}_init_calib_path",
            "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1277.npz",
        )

        self.zero_radius_px = int(
            device_ids.get(
                f"{self.name}_zero_radius_px",
                30,
            )
        )

        logging.info(f"Using DMD DLL: {self.dll_path}")
        logging.info(f"Using DMD device index: {self.device_index}")
        logging.info(f"Initial DMD state: {self.init_state}")
        logging.info(f"Initial calibration path: {self.init_calib_path}")

        # -----------------------------------------------------------------
        # Minimal calibration state.
        # The final chain file provides zero-order blocking and NV index points.
        # -----------------------------------------------------------------
        self.M_cam_to_dmd = None
        self.zero_dmd_xy = None
        self.zero_cam_xy = None
        self.loaded_dmd_points = None
        self.loaded_camera_points = None

        # Hardware state
        self.dmd = None
        self.connected = False

        # -----------------------------------------------------------------
        # Connect DMD hardware.
        # -----------------------------------------------------------------
        logging.info("Creating Dmd6500 object...")
        self.dmd = Dmd6500(str(self.dll_path))
        logging.info("Dmd6500 object created.")

        logging.info("Listing DMD devices...")
        n_devices = self.dmd.list_devices()
        logging.info(f"DMD devices found: {n_devices}")

        if n_devices <= self.device_index:
            raise RuntimeError(
                f"Requested DMD device index {self.device_index}, "
                f"but only {n_devices} device(s) found."
            )

        logging.info("Connecting to DMD...")
        self.dmd.connect(self.device_index)
        logging.info("DMD connect returned.")

        self.connected = True

        # -----------------------------------------------------------------
        # Load final chain calibration metadata.
        #
        # This does NOT apply a DMD mask for NVs.
        # It only loads:
        #   zero_dmd_xy
        #   M_cam_to_dmd
        #   loaded_dmd_points
        # into memory.
        # -----------------------------------------------------------------
        calib_loaded = False
        try:
            calib_loaded = self._load_calibration_file(self.init_calib_path)
        except Exception as exc:
            logging.warning(f"Could not load initial DMD calibration: {exc}")

        if calib_loaded:
            logging.info("Initial DMD final-chain calibration loaded.")

            if self.zero_dmd_xy is None:
                logging.warning(
                    "Loaded calibration has no zero_dmd_xy. "
                    "pass_zero_block will not block the zero order."
                )

            if self.M_cam_to_dmd is None:
                logging.warning(
                    "Loaded calibration has no M_cam_to_dmd."
                )

            if self.loaded_dmd_points is None:
                logging.warning(
                    "Loaded calibration has no dmd_points / pattern_dmd_points. "
                    "Index-based DMD control will not work."
                )
            else:
                logging.info(
                    f"Loaded {len(self.loaded_dmd_points)} DMD points for index control."
                )
        else:
            logging.warning(
                "No initial DMD calibration loaded. "
                "Use dmd.load_calibration(...) before index-based control."
            )

        # -----------------------------------------------------------------
        # Keep startup physically simple:
        # show true pass-all, no zero block yet.
        #
        # Later, call:
        #     dmd.initialize_pass_state()
        #
        # That will upload pass/block planes and apply pass_zero_block
        # using zero_dmd_xy from the final chain file.
        # -----------------------------------------------------------------
        self._show_pass_all(zero_block=True)

        logging.info(
            "DMD connected. Final-chain calibration loaded if available. "
            "Physical DMD left in true pass-all state."
        )
        logging.info("DMD init complete.")

    def stopServer(self):
        if self.dmd is not None:
            try:
                self.dmd.disconnect()
            except Exception:
                pass

        self.dmd = None
        self.connected = False

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _require_dmd(self):
        if self.dmd is None or not self.connected:
            raise RuntimeError("DMD is not connected.")

    def _resolve_path(self, path):
        p = Path(path)
        if p.is_absolute():
            return p
        return self.repo_path / p

    def _blank_mask(self, value):
        return np.full((DMD_HEIGHT, DMD_WIDTH), value, dtype=np.uint8)

    def _apply_disks(self, mask, coords, radius_px, value):
        """
        coords should be Nx2 DMD coordinates [x, y].
        """
        if coords is None:
            return mask

        coords = np.asarray(coords, dtype=np.float32)

        if coords.size == 0:
            return mask

        coords = coords.reshape(-1, 2)
        yy, xx = np.ogrid[:DMD_HEIGHT, :DMD_WIDTH]
        radius_px = int(radius_px)

        for x, y in coords:
            rr2 = (xx - float(x)) ** 2 + (yy - float(y)) ** 2
            mask[rr2 <= radius_px**2] = value

        return mask

    def _apply_zero_block(self, mask):
        """
        ON-pass mode:
            pass  = white / 255
            block = black / 0

        Therefore zero-order block is a black disk on a white/pass background.
        """
        if self.zero_dmd_xy is None:
            return mask

        return self._apply_disks(
            mask=mask,
            coords=np.array([self.zero_dmd_xy], dtype=np.float32),
            radius_px=self.zero_radius_px,
            value=self.BLOCK_VALUE,
        )

    def _send_and_show(self, plane, mask):
        self._require_dmd()
        mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.dmd.send_binary_plane(int(plane), mask)
        self.dmd.show_plane(int(plane))

    def _upload_basic_planes(self):
        self._require_dmd()

        pass_mask = self._blank_mask(self.PASS_VALUE)
        block_mask = self._blank_mask(self.BLOCK_VALUE)

        self.dmd.send_binary_plane(self.PASS_PLANE, pass_mask)
        self.dmd.send_binary_plane(self.BLOCK_PLANE, block_mask)

    def _show_pass_all(self, zero_block=True):
        mask = self._blank_mask(self.PASS_VALUE)

        if zero_block:
            mask = self._apply_zero_block(mask)

        self._send_and_show(self.PASS_PLANE, mask)

    def _show_block_all(self):
        mask = self._blank_mask(self.BLOCK_VALUE)
        self._send_and_show(self.BLOCK_PLANE, mask)

    def _load_calibration_file(self, calib_path):
        """
        Internal calibration loader.
        Used by initServer and load_calibration setting.
        """
        path = self._resolve_path(calib_path)

        if not path.exists():
            logging.warning(f"DMD calibration file not found: {path}")
            return False

        data = np.load(path, allow_pickle=True)

        if "M_cam_to_dmd" in data.files:
            self.M_cam_to_dmd = np.asarray(data["M_cam_to_dmd"], dtype=np.float32)

        if "zero_dmd_xy" in data.files:
            self.zero_dmd_xy = np.asarray(data["zero_dmd_xy"], dtype=np.float32).reshape(2)

        if "zero_cam_xy" in data.files:
            self.zero_cam_xy = np.asarray(data["zero_cam_xy"], dtype=np.float32).reshape(2)

        for key in ["pattern_dmd_points", "circle_dmd_points", "dmd_points"]:
            if key in data.files:
                self.loaded_dmd_points = np.asarray(data[key], dtype=np.float32)
                break

        for key in ["pattern_camera_points", "circle_camera_points", "slm_camera_points"]:
            if key in data.files:
                self.loaded_camera_points = np.asarray(data[key], dtype=np.float32)
                break

        logging.info(f"Loaded DMD calibration from: {path}")
        logging.info(f"zero_dmd_xy: {self.zero_dmd_xy}")
        logging.info(f"zero_cam_xy: {self.zero_cam_xy}")

        return True

    def _parse_coords_json(self, coords_json):
        coords = json.loads(coords_json)
        coords = np.asarray(coords, dtype=np.float32)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coordinates must be JSON Nx2 list: [[x, y], ...]")

        return coords

    def _parse_indices_json(self, indices_json):
        if indices_json.strip().lower() == "all":
            if self.loaded_dmd_points is None:
                raise RuntimeError("No loaded DMD points.")
            return np.arange(len(self.loaded_dmd_points), dtype=int)

        indices = json.loads(indices_json)
        return np.asarray(indices, dtype=int)

    def _require_calibration(self):
        if self.M_cam_to_dmd is None:
            raise RuntimeError("No camera-to-DMD affine calibration loaded.")

    def _map_camera_to_dmd(self, cam_pts):
        self._require_calibration()

        cam_pts = np.asarray(cam_pts, dtype=np.float32)

        if cam_pts.ndim != 2 or cam_pts.shape[1] != 2:
            raise ValueError("Camera points must be Nx2.")

        ones = np.ones((len(cam_pts), 1), dtype=np.float32)
        cam_h = np.hstack([cam_pts, ones])
        dmd_pts = cam_h @ self.M_cam_to_dmd.T

        return dmd_pts.astype(np.float32)
    
    def _upload_basic_planes(self):
        self._require_dmd()

        logging.info("_upload_basic_planes: building pass mask.")
        pass_mask = self._blank_mask(self.PASS_VALUE)

        logging.info("_upload_basic_planes: sending pass plane.")
        self.dmd.send_binary_plane(self.PASS_PLANE, pass_mask)
        logging.info("_upload_basic_planes: pass plane sent.")

        logging.info("_upload_basic_planes: building block mask.")
        block_mask = self._blank_mask(self.BLOCK_VALUE)

        logging.info("_upload_basic_planes: sending block plane.")
        self.dmd.send_binary_plane(self.BLOCK_PLANE, block_mask)
        logging.info("_upload_basic_planes: block plane sent.")

    # -----------------------------------------------------------------
    # Basic settings
    # -----------------------------------------------------------------
    
    @setting(0, returns="s")
    def get_state(self, c):
        """
        Return DMD server state.

        This is intentionally lightweight:
            - shows whether final chain calibration is loaded
            - shows whether zero-order blocking is available
            - shows whether NV-index DMD control is available
        """
        num_loaded_dmd_points = (
            0 if self.loaded_dmd_points is None else len(self.loaded_dmd_points)
        )

        num_loaded_camera_points = (
            0 if self.loaded_camera_points is None else len(self.loaded_camera_points)
        )

        resolved_init_calib_path = str(self._resolve_path(self.init_calib_path))

        state = {
            "name": self.name,
            "pc_name": self.pc_name,
            "connected": bool(self.connected),

            "DMD_WIDTH": DMD_WIDTH,
            "DMD_HEIGHT": DMD_HEIGHT,
            "PASS_VALUE": int(self.PASS_VALUE),
            "BLOCK_VALUE": int(self.BLOCK_VALUE),
            "convention": "ON/white=PASS, OFF/black=BLOCK",

            # Startup settings from config.
            "init_state": self.init_state,
            "init_calib_path": self.init_calib_path,
            "resolved_init_calib_path": resolved_init_calib_path,
            "zero_radius_px": int(self.zero_radius_px),

            # Loaded calibration status.
            "has_affine": self.M_cam_to_dmd is not None,
            "has_zero_dmd_xy": self.zero_dmd_xy is not None,
            "has_zero_cam_xy": self.zero_cam_xy is not None,
            "has_loaded_dmd_points": self.loaded_dmd_points is not None,

            "zero_dmd_xy": (
                None if self.zero_dmd_xy is None else self.zero_dmd_xy.tolist()
            ),
            "zero_cam_xy": (
                None if self.zero_cam_xy is None else self.zero_cam_xy.tolist()
            ),

            "num_loaded_dmd_points": num_loaded_dmd_points,
            "num_loaded_camera_points": num_loaded_camera_points,

            # Practical readiness flags.
            "can_zero_block": self.zero_dmd_xy is not None,
            "can_index_control": self.loaded_dmd_points is not None,
        }

        return json.dumps(state)

    @setting(1)
    def reset(self, c):
        """
        Default/bypass state.

        Since the DMD is only used for specific gated experiments,
        reset should leave the readout path open.

        ON-pass convention:
            white / 255 = pass
        """
        # self._require_dmd()
        # self._show_pass_all(zero_block=True)
        pass
        
    @setting(2, zero_block="b")
    def pass_all(self, c, zero_block=True):
        """
        Show all-pass plane.

        ON-pass convention:
            white / 255 = pass

        zero_block=True:
            pass all but keep 0th order blocked if zero_dmd_xy is loaded.

        zero_block=False:
            true bypass/pass-through mode.
        """
        self._require_dmd()
        self._show_pass_all(zero_block=bool(zero_block))

    @setting(3)
    def block_all(self, c):
        """
        Show all-block plane.
        """
        self._require_dmd()
        self._show_block_all()

    @setting(4, plane="i")
    def show_plane(self, c, plane):
        self._require_dmd()
        self.dmd.show_plane(int(plane))

    @setting(5)
    def upload_basic_planes(self, c):
        self._require_dmd()
        self._upload_basic_planes()
    
    @setting(6, returns="s")
    def initialize_pass_state(self, c):
        """
        Upload basic DMD planes and apply requested startup state.

        Minimal behavior:
            1. Require DMD hardware.
            2. Reload final chain calibration file.
            3. Upload pass/block planes.
            4. Apply init_state:
                - pass
                - pass_zero_block
                - block

        For normal operation, config should set:
            dmd_DLP6500_init_state = "pass_zero_block"
            dmd_DLP6500_init_calib_path =
                "dmdsuite/calibration/nv_chain_nuvu_thorcamDMD_dmd_1277.npz"
        """
        self._require_dmd()

        logging.info("initialize_pass_state called.")
        logging.info(f"Using init_state: {self.init_state}")
        logging.info(f"Using init_calib_path: {self.init_calib_path}")

        # -----------------------------------------------------------------
        # 1. Reload final chain calibration.
        #
        # This lets you regenerate the chain file and call initialize_pass_state()
        # again without restarting the server.
        # -----------------------------------------------------------------
        calib_loaded = False
        try:
            calib_loaded = self._load_calibration_file(self.init_calib_path)
        except Exception as exc:
            logging.warning(f"Could not load DMD calibration: {exc}")

        # -----------------------------------------------------------------
        # 2. Upload basic pass/block planes.
        # -----------------------------------------------------------------
        logging.info("Uploading basic PASS/BLOCK planes...")
        self._upload_basic_planes()
        logging.info("Basic PASS/BLOCK planes uploaded.")

        # -----------------------------------------------------------------
        # 3. Apply requested startup state.
        # -----------------------------------------------------------------
        shown_state = None

        if self.init_state == "pass":
            logging.info("Showing true pass-all.")
            self._show_pass_all(zero_block=False)
            shown_state = "pass"

        elif self.init_state == "pass_zero_block":
            if self.zero_dmd_xy is None:
                logging.warning(
                    "init_state is pass_zero_block, but zero_dmd_xy is not loaded. "
                    "Showing true pass-all instead."
                )
                self._show_pass_all(zero_block=False)
                shown_state = "pass_no_zero_loaded"
            else:
                logging.info("Showing pass-all with zero-order blocked.")
                self._show_pass_all(zero_block=True)
                shown_state = "pass_zero_block"

        elif self.init_state == "block":
            logging.info("Showing block-all.")
            self._show_block_all()
            shown_state = "block"

        else:
            logging.warning(
                f"Unknown init_state={self.init_state}. Showing true pass-all."
            )
            self._show_pass_all(zero_block=False)
            shown_state = "pass_unknown_init_state"

        # -----------------------------------------------------------------
        # 4. Return useful status.
        # -----------------------------------------------------------------
        out = {
            "calib_loaded": bool(calib_loaded),
            "init_state": self.init_state,
            "shown_state": shown_state,
            "init_calib_path": self.init_calib_path,
            "resolved_init_calib_path": str(self._resolve_path(self.init_calib_path)),

            "has_affine": self.M_cam_to_dmd is not None,
            "has_zero_dmd_xy": self.zero_dmd_xy is not None,
            "has_loaded_dmd_points": self.loaded_dmd_points is not None,

            "zero_dmd_xy": (
                None if self.zero_dmd_xy is None else self.zero_dmd_xy.tolist()
            ),
            "zero_radius_px": int(self.zero_radius_px),

            "num_loaded_dmd_points": (
                0 if self.loaded_dmd_points is None else len(self.loaded_dmd_points)
            ),
        }

        return json.dumps(out)
    # -----------------------------------------------------------------
    # Zero-order control
    # -----------------------------------------------------------------

    @setting(10, zero_xy_json="s", radius_px="i", returns="s")
    def set_zero_block(self, c, zero_xy_json, radius_px=30):
        """
        Store 0th-order DMD coordinate and radius, then show pass+zero-block.

        Example:
            dmd.set_zero_block("[887, 460]", 30)
        """
        self._require_dmd()

        xy = np.asarray(json.loads(zero_xy_json), dtype=np.float32).reshape(2)

        self.zero_dmd_xy = xy
        self.zero_radius_px = int(radius_px)

        self._show_pass_all(zero_block=True)

        return json.dumps(
            {
                "zero_dmd_xy": self.zero_dmd_xy.tolist(),
                "zero_radius_px": self.zero_radius_px,
            }
        )

    @setting(11)
    def zero_block_on(self, c):
        """
        Show pass-all with zero-order blocked.
        """
        self._require_dmd()
        self._show_pass_all(zero_block=True)

    @setting(12)
    def zero_block_off(self, c):
        """
        Show true pass-all without zero-order block.
        """
        self._require_dmd()
        self._show_pass_all(zero_block=False)

    @setting(13, radius_px="i")
    def set_zero_radius(self, c, radius_px):
        self.zero_radius_px = int(radius_px)

    # -----------------------------------------------------------------
    # Calibration loading / mapping
    # -----------------------------------------------------------------

    @setting(20, calib_path="s", returns="s")
    def load_calibration(self, c, calib_path):
        """
        Load saved affine calibration file.

        Expected keys:
            M_cam_to_dmd
            zero_dmd_xy

        Optional keys:
            zero_cam_xy
            pattern_dmd_points / circle_dmd_points / dmd_points
            pattern_camera_points / circle_camera_points / slm_camera_points
        """
        self._require_dmd()

        ok = self._load_calibration_file(calib_path)

        if ok:
            self._show_pass_all(zero_block=True)

        out = {
            "loaded": bool(ok),
            "path": str(self._resolve_path(calib_path)),
            "has_affine": self.M_cam_to_dmd is not None,
            "zero_dmd_xy": None if self.zero_dmd_xy is None else self.zero_dmd_xy.tolist(),
            "zero_cam_xy": None if self.zero_cam_xy is None else self.zero_cam_xy.tolist(),
            "num_loaded_dmd_points": (
                0 if self.loaded_dmd_points is None else len(self.loaded_dmd_points)
            ),
        }

        return json.dumps(out)

    @setting(21, cam_coords_json="s", returns="s")
    def map_camera_to_dmd(self, c, cam_coords_json):
        """
        Map camera coordinates to DMD coordinates using loaded affine.

        Example:
            [[650, 360], [800, 360], [725, 520]]
        """
        cam_pts = self._parse_coords_json(cam_coords_json)
        dmd_pts = self._map_camera_to_dmd(cam_pts)
        return json.dumps(dmd_pts.tolist())

    @setting(22, dmd_coords_json="s", returns="s")
    def load_dmd_points(self, c, dmd_coords_json):
        """
        Store DMD coordinates for index-based spot control.
        """
        self.loaded_dmd_points = self._parse_coords_json(dmd_coords_json)
        return json.dumps({"num_loaded_dmd_points": len(self.loaded_dmd_points)})

    @setting(23, cam_coords_json="s", returns="s")
    def load_camera_points_as_dmd_points(self, c, cam_coords_json):
        """
        Map camera points to DMD points using loaded affine and store them.
        """
        cam_pts = self._parse_coords_json(cam_coords_json)
        dmd_pts = self._map_camera_to_dmd(cam_pts)

        self.loaded_camera_points = cam_pts
        self.loaded_dmd_points = dmd_pts

        return json.dumps(
            {
                "num_loaded_dmd_points": len(self.loaded_dmd_points),
                "dmd_points": self.loaded_dmd_points.tolist(),
            }
        )

    # -----------------------------------------------------------------
    # Direct coordinate mask control
    # -----------------------------------------------------------------

    @setting(30, dmd_coords_json="s", radius_px="i", plane="i")
    def pass_spots(self, c, dmd_coords_json, radius_px=20, plane=230):
        """
        Block all background, pass only selected DMD spots.

        ON-pass:
            background = black/block
            selected disks = white/pass
        """
        self._require_dmd()

        coords = self._parse_coords_json(dmd_coords_json)

        mask = self._blank_mask(self.BLOCK_VALUE)
        mask = self._apply_disks(mask, coords, radius_px, self.PASS_VALUE)
        mask = self._apply_zero_block(mask)

        self._send_and_show(plane, mask)

    @setting(31, dmd_coords_json="s", radius_px="i", plane="i")
    def block_spots(self, c, dmd_coords_json, radius_px=20, plane=230):
        """
        Pass all background, block selected DMD spots.

        ON-pass:
            background = white/pass
            selected disks = black/block
        """
        self._require_dmd()

        coords = self._parse_coords_json(dmd_coords_json)

        mask = self._blank_mask(self.PASS_VALUE)
        mask = self._apply_disks(mask, coords, radius_px, self.BLOCK_VALUE)
        mask = self._apply_zero_block(mask)

        self._send_and_show(plane, mask)

    # -----------------------------------------------------------------
    # Index-based control using loaded DMD points
    # -----------------------------------------------------------------

    @setting(40, indices_json="s", radius_px="i", plane="i")
    def pass_loaded_indices(self, c, indices_json, radius_px=20, plane=230):
        """
        Pass only selected spots from loaded_dmd_points.

        Example:
            pass_loaded_indices("[0, 1, 2]", 20, 230)
            pass_loaded_indices("all", 20, 230)
        """
        self._require_dmd()

        if self.loaded_dmd_points is None:
            raise RuntimeError(
                "No loaded DMD points. Use load_dmd_points or "
                "load_camera_points_as_dmd_points."
            )

        inds = self._parse_indices_json(indices_json)
        coords = self.loaded_dmd_points[inds]

        mask = self._blank_mask(self.BLOCK_VALUE)
        mask = self._apply_disks(mask, coords, radius_px, self.PASS_VALUE)
        mask = self._apply_zero_block(mask)

        self._send_and_show(plane, mask)

    @setting(41, indices_json="s", radius_px="i", plane="i")
    def block_loaded_indices(self, c, indices_json, radius_px=20, plane=230):
        """
        Pass all background, block selected loaded spots.

        Example:
            block_loaded_indices("[0]", 20, 230)
        """
        self._require_dmd()

        if self.loaded_dmd_points is None:
            raise RuntimeError(
                "No loaded DMD points. Use load_dmd_points or "
                "load_camera_points_as_dmd_points."
            )

        inds = self._parse_indices_json(indices_json)
        coords = self.loaded_dmd_points[inds]

        mask = self._blank_mask(self.PASS_VALUE)
        mask = self._apply_disks(mask, coords, radius_px, self.BLOCK_VALUE)
        mask = self._apply_zero_block(mask)

        self._send_and_show(plane, mask)

    @setting(42, radius_px="i", plane="i")
    def pass_all_loaded(self, c, radius_px=20, plane=230):
        """
        Block background and pass all loaded DMD points.
        Useful when using the DMD as a spot-only gate.
        """
        self._require_dmd()

        if self.loaded_dmd_points is None:
            raise RuntimeError("No loaded DMD points.")

        mask = self._blank_mask(self.BLOCK_VALUE)
        mask = self._apply_disks(
            mask,
            self.loaded_dmd_points,
            radius_px,
            self.PASS_VALUE,
        )
        mask = self._apply_zero_block(mask)

        self._send_and_show(plane, mask)

    @setting(43, radius_px="i", plane="i")
    def block_all_loaded(self, c, radius_px=20, plane=230):
        """
        Pass background and block all loaded DMD points.
        """
        self._require_dmd()

        if self.loaded_dmd_points is None:
            raise RuntimeError("No loaded DMD points.")

        mask = self._blank_mask(self.PASS_VALUE)
        mask = self._apply_disks(
            mask,
            self.loaded_dmd_points,
            radius_px,
            self.BLOCK_VALUE,
        )
        mask = self._apply_zero_block(mask)

        self._send_and_show(plane, mask)


    @setting(50, axis="s", pos="v[]", width="i", plane="i")
    def show_blocking_stripe(self, c, axis, pos, width=40, plane=220):
        """
        ON-pass calibration stripe.

        Current convention:
            white / 255 = pass
            black / 0   = block

        This creates a white pass-all background and adds one black
        blocking stripe.
        """
        self._require_dmd()

        axis = axis.lower()
        pos = int(round(pos))
        width = int(width)

        mask = self._blank_mask(self.PASS_VALUE)

        if axis == "x":
            x0 = max(0, pos - width // 2)
            x1 = min(DMD_WIDTH, pos + width // 2)
            mask[:, x0:x1] = self.BLOCK_VALUE

        elif axis == "y":
            y0 = max(0, pos - width // 2)
            y1 = min(DMD_HEIGHT, pos + width // 2)
            mask[y0:y1, :] = self.BLOCK_VALUE

        else:
            raise ValueError("axis must be 'x' or 'y'")

        mask = self._apply_zero_block(mask)
        self._send_and_show(plane, mask)


    @setting(51, x="v[]", y="v[]", radius_px="i", plane="i")
    def show_blocking_disk(self, c, x, y, radius_px=25, plane=230):
        """
        ON-pass single blocking disk.

        White background = pass.
        Black disk = block.
        """
        self._require_dmd()

        mask = self._blank_mask(self.PASS_VALUE)

        coords = np.array([[float(x), float(y)]], dtype=np.float32)
        mask = self._apply_disks(
            mask=mask,
            coords=coords,
            radius_px=radius_px,
            value=self.BLOCK_VALUE,
        )

        mask = self._apply_zero_block(mask)
        self._send_and_show(plane, mask)


    @setting(52, x="v[]", y="v[]", radius_px="i", plane="i")
    def show_pass_disk(self, c, x, y, radius_px=25, plane=231):
        """
        ON-pass selected pass disk.

        Black background = block.
        White disk = pass.
        """
        self._require_dmd()

        mask = self._blank_mask(self.BLOCK_VALUE)

        coords = np.array([[float(x), float(y)]], dtype=np.float32)
        mask = self._apply_disks(
            mask=mask,
            coords=coords,
            radius_px=radius_px,
            value=self.PASS_VALUE,
        )

        mask = self._apply_zero_block(mask)
        self._send_and_show(plane, mask)


    @setting(53, x="v[]", y="v[]", radius_px="i", returns="s")
    def update_zero_block_xy(self, c, x, y, radius_px=30):
        """
        Convenience setting for storing zero-order DMD coordinate.
        """
        self.zero_dmd_xy = np.array([float(x), float(y)], dtype=np.float32)
        self.zero_radius_px = int(radius_px)

        self._show_pass_all(zero_block=True)

        return json.dumps(
            {
                "zero_dmd_xy": self.zero_dmd_xy.tolist(),
                "zero_radius_px": self.zero_radius_px,
            }
        )

__server__ = DmdDlp6500()

if __name__ == "__main__":
    from labrad import util

    util.runServer(__server__)