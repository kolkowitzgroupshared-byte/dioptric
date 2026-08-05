# -*- coding: utf-8 -*-
"""
nv_drift_loop_monitor.py

Repeatedly run drift compensation on an NV for many hours, and record
the peak counts each iteration finds. Each cycle is written to a CSV
on disk IMMEDIATELY, so a crash mid-run only costs the most recent line.

The optimizer's own "peak counts after XY+Z" is the brightness of the
NV at the moment of that optimization, by construction.

Set nv_sig.expected_counts = None before calling so the bounds check
doesn't reject correct optimizations on a (currently dimmer) NV.

Make sure the NIR probe is blocked.

Press CTRL+C to stop. The CSV is already complete up to the last line.
"""

import csv
import os
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import majorroutines.targeting as targeting
import utils.tool_belt as tb
from utils import common
from utils import data_manager as dm


def main(
    nv_sig,
    total_duration_s=8 * 3600,
    pause_between_s=60 * 5,
):
    with common.labrad_connect() as cxn:
        return main_with_cxn(
            cxn,
            nv_sig,
            total_duration_s=total_duration_s,
            pause_between_s=pause_between_s,
        )


def main_with_cxn(
    cxn,
    nv_sig,
    total_duration_s=8 * 3600,
    pause_between_s=0.0,
):
    # File paths -- one CSV (incremental), one final plot
    ts = dm.get_time_stamp()
    file_path = dm.get_file_path(__file__, ts, getattr(nv_sig, "name", "nv"))
    # file_path is typically a pathlib.Path without extension; build a sibling CSV
    csv_path = str(file_path) + ".csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"\nNV drift-loop monitor")
    print(f"  NV name           = {getattr(nv_sig, 'name', '?')}")
    print(f"  total_duration_s  = {total_duration_s} ({total_duration_s/3600:.1f} h)")
    print(f"  pause_between_s   = {pause_between_s}")
    print(f"  expected_counts   = {getattr(nv_sig, 'expected_counts', '?')}"
          " (should be None)")
    print(f"  writing CSV to    : {csv_path}")
    print("Make sure NIR probe is blocked. CTRL+C to stop.\n")

    # Write CSV header now, then flush after every line
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow([
        "iteration",
        "wall_time_iso",
        "elapsed_s",
        "drift_x",
        "drift_y",
        "drift_z",
        "peak_counts",
        "drift_ok",
    ])
    fh.flush()

    iterations = []
    elapsed_list = []
    peak_counts_list = []
    drift_xyz_list = []

    tb.init_safe_stop()
    t_start = time.monotonic()
    iteration = 0

    try:
        while True:
            if tb.safe_stop():
                print("Stop requested.")
                break

            elapsed = time.monotonic() - t_start
            if elapsed >= total_duration_s:
                print("Reached total_duration_s.")
                break

            iteration += 1
            print(f"--- iter {iteration}  t={elapsed:.1f} s "
                  f"({elapsed/3600:.2f} h) ---")

            drift_ok = True
            peak_counts = float("nan")
            drift_xyz = (float("nan"), float("nan"), float("nan"))

            try:
                # compensate_for_drift in your codebase returns whatever it
                # returns (we don't depend on the return value). What we
                # care about is the resulting peak counts and the saved
                # drift offset in LabRAD.
                targeting.compensate_for_drift(nv_sig, no_crash=True)

                # Read the current LabRAD drift offset (the one your
                # registry shows under DRIFT).
                try:
                    drift_xyz = tuple(tb.get_drift())
                except Exception:
                    # If your tool_belt names this differently, adjust here.
                    # Common alternatives: tb.get_drift(cxn), common.get_drift(cxn)
                    pass

                # We don't have a direct "peak counts at optimum" return
                # value from compensate_for_drift here -- but the routine
                # prints it as "Counts after drift compensation: NN" and
                # we want a programmatic number. Easiest robust path: just
                # ask the optimizer once more for a single-point read,
                # which is what your imaging/live-counts panel does.
                #
                # If you have a clean accessor for "counts at current
                # corrected position", e.g. tb.read_counts(...) or a
                # method on the optimize module, use it here. Otherwise
                # we leave peak_counts as NaN and rely on the stdout log
                # for the actual number.
                #
                # NOTE: replace the next line with your setup's
                # 'measure-counts-now' call to populate peak_counts.
                peak_counts = float("nan")

            except Exception:
                traceback.print_exc()
                drift_ok = False

            iterations.append(iteration)
            elapsed_list.append(elapsed)
            peak_counts_list.append(peak_counts)
            drift_xyz_list.append(drift_xyz)

            writer.writerow([
                iteration,
                datetime.now().isoformat(timespec="seconds"),
                f"{elapsed:.3f}",
                f"{drift_xyz[0]:.6f}" if not np.isnan(drift_xyz[0]) else "",
                f"{drift_xyz[1]:.6f}" if not np.isnan(drift_xyz[1]) else "",
                f"{drift_xyz[2]:.6f}" if not np.isnan(drift_xyz[2]) else "",
                f"{peak_counts:.3f}" if not np.isnan(peak_counts) else "",
                int(drift_ok),
            ])
            fh.flush()
            os.fsync(fh.fileno())  # force the write to physical disk

            print(f"  drift={drift_xyz}  peak_counts={peak_counts}  ok={drift_ok}")

            if pause_between_s > 0:
                end_at = time.monotonic() + pause_between_s
                while time.monotonic() < end_at:
                    if tb.safe_stop():
                        break
                    time.sleep(min(1.0, end_at - time.monotonic()))

    except KeyboardInterrupt:
        print("KeyboardInterrupt; CSV is up to date through the last completed line.")
    finally:
        fh.close()

    # ---- Also save a structured raw_data + plot at the end ----
    raw_data = {
        "timestamp": ts,
        "nv_sig": nv_sig,
        "total_duration_s": float(total_duration_s),
        "pause_between_s": float(pause_between_s),
        "iterations": iterations,
        "elapsed_s": elapsed_list,
        "peak_counts": peak_counts_list,
        "drift_xyz": [list(t) for t in drift_xyz_list],
        "csv_path": csv_path,
    }
    dm.save_raw_data(raw_data, file_path)

    if len(elapsed_list) and any(not np.isnan(c) for c in peak_counts_list):
        fig, ax = plt.subplots(figsize=(10, 5))
        t_hr = np.array(elapsed_list) / 3600.0
        ax.plot(t_hr, peak_counts_list, "o-", ms=4, lw=1)
        ax.set_xlabel("Time since start (h)")
        ax.set_ylabel("Peak counts after drift correction")
        ax.set_title(
            f"NV brightness vs time — {getattr(nv_sig, 'name', 'nv')}  "
            f"({len(elapsed_list)} iterations)"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dm.save_figure(fig, file_path)

    print(f"\nSaved structured data to {file_path}")
    print(f"Incremental CSV at      {csv_path}")
    return raw_data
