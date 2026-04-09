# -*- coding: utf-8 -*-
"""
2D scan over a designated area, collecting photon counts at each point.
Generates an image of the sample in COUNTER mode.

Supports galvo (AOD) scanning via SEQUENCE or piezo scanning via STREAM/STEP.
"""

import time
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from utils import common
import utils.data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import (
    CollectionMode,
    CoordsKey,
    CountFormat,
    NVSig,
    PosControlMode,
    VirtualLaserKey,
)

SEQ_FILE_SEQUENCE_SCAN = "simple_readout_laser_free_test.py"  # expects [0, readout_ns, x[], y[]]
SEQ_FILE_PIXEL_READOUT  = "simple_readout.py"                 # per-pixel readout when we move in Python

# --- replace the snake helper with a raster helper ---
def _raster_fill(vals, img, state):
    """
    Row-major fill, bottom row first, left -> right on every row.
    `state` holds a single integer: number of pixels already written.
    """
    h, w = img.shape
    if not state:        # initialize write index
        state[:] = [0]
    i = state[0]
    for v in vals:
        if i >= h * w:
            break
        row = i // w
        col = i %  w
        y = (h - 1) - row   # bottom row first (origin='lower')
        x = col             # left -> right
        img[y, x] = v
        i += 1
    state[:] = [i]

def confocal_scan(nv_sig: NVSig, x_range, y_range, num_steps, nv_minus_init=False):
    cfg = common.get_config_dict()
    count_fmt: CountFormat = cfg["count_format"]

    positioner = pos.get_laser_positioner(VirtualLaserKey.IMAGING)
    mode = pos.get_positioner_control_mode(positioner)

    # Build grid (1D coordinate axes + random-access XY)
    x0, y0 = pos.get_nv_coords(nv_sig, coords_key=CoordsKey.PIXEL)
    X, Y, x1d, y1d, extent = pos.get_scan_grid_2d(x0, y0, x_range, y_range, num_steps, num_steps)
    h = w = num_steps
    total = h * w
    # --- UI throttling: update plot every N pixels (about ~8 updates/row) ---
    UPDATE_EVERY = 1
    pixels_done = 0
    # Timing / laser
    vld = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(vld["duration"])))
    readout_s = readout_ns / 1e9
    readout_laser = vld["physical_name"]

    pulse = tb.get_server_pulse_streamer()
    ctr   = tb.get_server_counter()


    img = np.full((h, w), np.nan, float)
    img_kcps = np.copy(img) if count_fmt == CountFormat.KCPS else None

    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Voltage (V)")
    kpl.imshow(ax, img if img_kcps is None else img_kcps,
            #    title=f"XY={nv_sig.coords[CoordsKey.SAMPLE]}, Z={nv_sig.coords[CoordsKey.Z]}, {ccryo.get_sample_name()}",
               title=f"{readout_laser}, {readout_ns/1e6:.1f} ms",
               cbar_label=("Kcps" if img_kcps is not None else "Raw Counts"),
               extent=extent)

    tb.reset_cfm()
    tb.init_safe_stop()
    ctr.start_tag_stream()

    # ---- sequence loading (hardware delay, not Python sleep) ----
    pos_key = CoordsKey.PIXEL
    delay_ns = int(cfg["Positioning"]["Positioners"][pos_key]["delay"])  # e.g. 400e3 ns for galvo
    # period_ns = pulse.stream_load(
    #     SEQ_FILE_PIXEL_READOUT,
    #     tb.encode_seq_args([delay_ns, readout_ns, readout_laser, 1.0])  # laser args ignored by your seq
    # )[0]

    readout_vkey = VirtualLaserKey.IMAGING

    period_ns = pulse.stream_load(
        "simple_readout.py",
        tb.encode_seq_args([delay_ns, readout_ns, readout_vkey, 100, 1])  # 100ns marker at readout start
    )[0]

    try:
        if mode == PosControlMode.SEQUENCE:
            # ---- RASTER ORDER COORDINATE LISTS (no serpentine) ----
            Xr = []
            Yr = []
            for row in range(h):
                y = y1d[(h - 1) - row]
                # y = y1d[row]   # bottom row first
                for col in range(w):
                    x = x1d[col]        # left -> right always
                    Xr.append(x); Yr.append(y)

            seq_args = [0, readout_ns, list(Xr), list(Yr)]   # delay handled in pixel-readout mode only
            period_ns = pulse.stream_load(SEQ_FILE_PIXEL_READOUT, tb.encode_seq_args(seq_args))[0]
            pulse.stream_start(total)

            written = []  # state for _raster_fill
            done = 0
            # conservative timeout
            timeout = time.time() + (period_ns * 1e-9) * total + 10.0

            while done < total and not tb.safe_stop():
                if time.time() > timeout: break
                if nv_minus_init:
                    raw = ctr.read_counter_modulo_gates(2)
                    vals = [max(int(a)-int(b), 0) for (a,b) in raw]
                else:
                    raw = ctr.read_counter_simple()
                    vals = [int(v) for v in raw]

                if not vals:
                    continue

                _raster_fill(vals, img, written)
                done = written[0]

                if img_kcps is not None:
                    img_kcps[:] = (img/1000.0)/readout_s
                    kpl.imshow_update(ax, img_kcps)
                else:
                    kpl.imshow_update(ax, img)

        else:
            # ---- STEP/STREAM CONTROL: drive stage in raster order ----
            # pulse.stream_start(total)

            written = []  # state for _raster_fill
            for row in range(h):
                if tb.safe_stop(): break
                y = y1d[(h - 1) - row]  # bottom row first
                for col in range(w):
                    if tb.safe_stop(): break
                    x = x1d[col]        # left -> right
                    pos.set_xyz((x, y), positioner=positioner)
                    
                    pulse.stream_start(1) 
                    # read exactly ONE sample per pixel
                    if nv_minus_init:
                        raw = ctr.read_counter_modulo_gates(2, 1)  # [[a,b]]
                        vals = [max(int(raw[0][0]) - int(raw[0][1]), 0)]
                    else:
                        raw = ctr.read_counter_simple(1)            # [c]
                        vals = [int(raw[0])]

                    if not vals:
                        continue

                    _raster_fill(vals, img, written)
                        
                    # UI throttle
                    pixels_done = written[0]
                    if (pixels_done % UPDATE_EVERY) == 0:
                        if img_kcps is not None:
                            img_kcps[:] = (img / 1000.0) / readout_s
                            kpl.imshow_update(ax, img_kcps)
                        else:
                            kpl.imshow_update(ax, img)

    finally:
        ctr.clear_buffer()
        tb.reset_cfm()

    # ---- Save (units consistent with display/return) ----
    is_kcps = (count_fmt == CountFormat.KCPS)
    img_out = (img/1000.0)/readout_s if is_kcps else img
    units_out = "kcps" if is_kcps else "counts"

    ts = dm.get_time_stamp()
    raw = {
    "timestamp": ts,
    "nv_sig": nv_sig,  
    "num_steps": num_steps,
    "x_range": x_range, "y_range": y_range,
    "x_center": x0, "y_center": y0,
    "extent": extent,
    "readout_ns": readout_ns, "readout_units": "ns",
    "img_array": img_out.astype(float), "img_array_units": units_out,
    "x_coords_1d": x1d, "y_coords_1d": y1d,
    }
    path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    dm.save_raw_data(raw, path)
    dm.save_figure(fig, path)
    kpl.show()
    return img_out, x1d, y1d

# # ---------------------------------------------------------------------------
# Simple viewer for previously saved scan files (optional)
# ---------------------------------------------------------------------------
def get_coord(coords, key):
    # Allow either Enum or string
    if hasattr(key, "name"):  # Enum
        return coords.get(key, coords.get(key.name))
    return coords.get(key)

if __name__ == "__main__":
    file_name = "2026_04_08-15_51_15-(lovelace)"

    data = dm.get_raw_data(file_name)
    print("Top-level keys in saved file:")
    print(list(data.keys()))
    # nv_sig = data["nv_sig"]
    # z_coord = nv_sig["coords"]["z"]
    # z_coord = nv_sig.coords[CoordsKey.Z]

    # print(nv_sig)
    # z_coord = get_coord(nv_sig.coords, CoordsKey.Z)
    # print(z_coord)
    img_array = np.array(data["img_array"])
    readout_ns = data["readout_ns"]
    img_array_kcps = (img_array / 1000.0) / (readout_ns * 1e-9)
    extent = data.get("extent", None)

    kpl.init_kplotlib()
    fig, ax = plt.subplots()
    _ = kpl.imshow(ax, img_array, cbar_label="Counts", extent=extent)
    ax.set_title(data.get("title", "Saved scan"))
    plt.show(block=True)
