# -*- coding: utf-8 -*-
"""
XY optimization routine using grid/raster scan pattern.

Scans the galvo in a small grid around the current position,
collects photon counts, and fits a 2D Gaussian to find the optimal XY position.

Uses the same raster scan pattern as do_image_sample for reliable Gaussian fitting.

Created on December 16th, 2025

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


def gaussian_2d(coords, amplitude, x0, y0, sigma, offset):
    """
    2D Gaussian function for fitting.

    Parameters
    ----------
    coords : tuple
        (x, y) coordinate arrays
    amplitude : float
        Peak height above offset
    x0, y0 : float
        Center coordinates
    sigma : float
        Standard deviation (same for x and y)
    offset : float
        Background offset

    Returns
    -------
    np.ndarray
        Gaussian values at each coordinate
    """
    x, y = coords
    return offset + amplitude * np.exp(
        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    )


def main(
    nv_sig: NVSig,
    num_steps: int = 15,
    scan_range: float = None,
    fit_method: str = "gaussian",
    move_to_optimal: bool = True,
    save_data: bool = True,
    num_averages: int = 1,
) -> dict:
    """
    Optimize XY position using a grid/raster scan pattern.

    Scans the galvo in a small grid around the current position,
    collects photon counts, and fits a 2D Gaussian to find the optimal
    XY coordinates.

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    num_steps : int, optional
        Number of steps per axis (creates num_steps x num_steps grid). Default: 15
    scan_range : float, optional
        Total scan range in volts. If None, uses config's optimize_range.
        Step size = scan_range / (num_steps - 1). Default: None
    fit_method : str, optional
        Method to find optimal position: "gaussian" for 2D Gaussian fit,
        "max_counts" for maximum counts. Default: "gaussian"
    move_to_optimal : bool, optional
        Whether to move galvo to optimal position after fitting. Default: True
    save_data : bool, optional
        Whether to save data and plot. Default: True
    num_averages : int, optional
        Number of counts to average at each position. Default: 1

    Returns
    -------
    dict
        Results containing:
        - opti_x: Optimal X coordinate
        - opti_y: Optimal Y coordinate
        - opti_counts: Counts at optimal position
        - img_array: 2D array of counts
        - x_vals: Array of X coordinates
        - y_vals: Array of Y coordinates
        - fit_params: Gaussian fit parameters (if fit_method="gaussian")
        - fit_success: Whether fit succeeded
        - initial_coords: Initial [x, y] coordinates
    """

    ### Setup

    config = common.get_config_dict()

    # Get hardware servers
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    # Get optimize range - use provided value or fall back to config
    if scan_range is None:
        optimize_range = pos.get_positioner_optimize_range(CoordsKey.PIXEL)
    else:
        optimize_range = scan_range

    # Setup laser for imaging (matching confocal_image_sample.py)
    laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(laser_dict["duration"]))
    )
    readout_s = readout_ns / 1e9
    laser_name = laser_dict["physical_name"]

    # Get initial position
    initial_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
    center_x = initial_coords[0]
    center_y = initial_coords[1]

    # Calculate step size
    step_size = optimize_range / (num_steps - 1) if num_steps > 1 else 0

    print(f"\nXY Optimization - Grid Scan")
    print(f"="*50)
    print(f"Initial position: X={center_x:.4f}, Y={center_y:.4f}")
    print(f"Scan range: {optimize_range:.4f} V")
    print(f"Step size: {step_size:.6f} V")
    print(f"Grid: {num_steps} x {num_steps} = {num_steps**2} points")
    print(f"Fit method: {fit_method}")
    print(f"="*50 + "\n")

    ### Generate grid coordinates (same pattern as do_image_sample)

    x_vals = np.linspace(center_x - optimize_range/2, center_x + optimize_range/2, num_steps)
    y_vals = np.linspace(center_y - optimize_range/2, center_y + optimize_range/2, num_steps)

    # Create meshgrid for fitting later
    X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

    total_points = num_steps * num_steps
    print(f"Total scan points: {total_points}")

    ### Setup figure for real-time display (confocal-style imshow)

    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize image array with NaN
    img_array = np.full((num_steps, num_steps), np.nan)

    # Set up extent for imshow (left, right, bottom, top)
    half_pixel_x = (x_vals[1] - x_vals[0]) / 2 if num_steps > 1 else optimize_range / 4
    half_pixel_y = (y_vals[1] - y_vals[0]) / 2 if num_steps > 1 else optimize_range / 4
    extent = [
        x_vals[0] - half_pixel_x,
        x_vals[-1] + half_pixel_x,
        y_vals[-1] + half_pixel_y,  # Note: y ordering for image display
        y_vals[0] - half_pixel_y,
    ]

    # Initial imshow display
    kpl.imshow(
        ax,
        img_array,
        title=f"XY Optimization - {laser_name}, {readout_ns/1e6:.1f} ms",
        x_label="X Voltage (V)",
        y_label="Y Voltage (V)",
        cbar_label="Counts",
        extent=extent,
    )

    # Mark initial position
    ax.plot(center_x, center_y, "r+", markersize=15, markeredgewidth=2, zorder=10)

    plt.ion()
    plt.pause(0.1)

    ### Hardware setup (exact order from confocal_image_sample.py)

    tb.reset_cfm()
    tb.init_safe_stop()
    counter.start_tag_stream()  # Start BEFORE loading sequence

    # Load pulse sequence
    seq_file = "simple_readout.py"
    positioner_dict = config["Positioning"]["Positioners"][CoordsKey.PIXEL]
    delay_ns = int(positioner_dict.get("delay", 0))

    seq_args = [delay_ns, readout_ns, laser_name, 1.0]
    pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args))

    ### Collect counts using raster scan (same pattern as confocal_image_sample.py)

    pixel_count = 0

    try:
        print(f"Scanning {total_points} positions (raster scan)...")

        # Raster scan: row by row, bottom to top, left to right
        for row in range(num_steps):
            if tb.safe_stop():
                print("\n[STOPPED] User interrupt")
                break

            y = y_vals[(num_steps - 1) - row]  # Bottom row first (origin='lower')

            for col in range(num_steps):
                if tb.safe_stop():
                    break

                x = x_vals[col]  # Left to right

                # Move galvo to this position
                pos.set_xyz((x, y), positioner=CoordsKey.PIXEL)

                # Collect samples at this position
                samples = []
                for _ in range(num_averages):
                    pulse_gen.stream_start(1)
                    raw = counter.read_counter_simple(1)
                    if raw and len(raw) > 0:
                        samples.append(int(raw[0]))

                # Average and store in image array
                avg_counts = np.mean(samples) if samples else 0

                # Store in image array (row index from bottom)
                img_row = (num_steps - 1) - row
                img_array[img_row, col] = avg_counts
                pixel_count += 1

            # Update display after each row
            kpl.imshow_update(ax, img_array)
            plt.pause(0.01)

            # Progress output
            print(f"  Row {row+1}/{num_steps} complete, last count: {avg_counts:.0f}")

    finally:
        counter.clear_buffer()
        tb.reset_cfm()
        tb.reset_safe_stop()

    print(f"\nCollected {pixel_count} data points")

    ### Find optimal position

    opti_x = None
    opti_y = None
    fit_params = None
    fit_success = False

    # Flatten the image array for fitting
    img_flat = img_array.flatten()
    x_flat = X_mesh.flatten()
    y_flat = Y_mesh.flatten()

    # Remove NaN values
    valid_mask = ~np.isnan(img_flat)
    img_valid = img_flat[valid_mask]
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]

    if fit_method == "gaussian" and len(img_valid) >= 10:
        # 2D Gaussian fit
        try:
            # Initial guesses from data
            max_idx = np.argmax(img_valid)
            x0_guess = x_valid[max_idx]
            y0_guess = y_valid[max_idx]
            offset_guess = np.nanmin(img_array)
            amplitude_guess = np.nanmax(img_array) - offset_guess
            sigma_guess = optimize_range / 4

            guess = [amplitude_guess, x0_guess, y0_guess, sigma_guess, offset_guess]
            bounds = (
                [0, x_vals[0], y_vals[0], 0, 0],
                [np.inf, x_vals[-1], y_vals[-1], optimize_range, np.inf]
            )

            popt, _ = curve_fit(
                gaussian_2d,
                (x_valid, y_valid),
                img_valid,
                p0=guess,
                bounds=bounds,
                maxfev=10000
            )

            opti_x = popt[1]
            opti_y = popt[2]
            fit_params = {
                "amplitude": popt[0],
                "x0": popt[1],
                "y0": popt[2],
                "sigma": popt[3],
                "offset": popt[4],
            }
            fit_success = True

            print(f"\n2D Gaussian fit results:")
            print(f"  Optimal X: {opti_x:.4f} V")
            print(f"  Optimal Y: {opti_y:.4f} V")
            print(f"  Amplitude: {popt[0]:.0f} counts")
            print(f"  Sigma: {popt[3]:.4f} V")
            print(f"  Offset: {popt[4]:.0f} counts")

        except Exception as e:
            print(f"\nGaussian fit failed: {e}")
            print("Falling back to max counts method")
            fit_method = "max_counts"

    if fit_method == "max_counts" or (not fit_success and len(img_valid) > 0):
        # Use position of maximum counts
        max_idx = np.argmax(img_valid)
        opti_x = x_valid[max_idx]
        opti_y = y_valid[max_idx]
        print(f"\nMax counts method:")
        print(f"  Optimal X: {opti_x:.4f} V")
        print(f"  Optimal Y: {opti_y:.4f} V")
        print(f"  Max counts: {img_valid[max_idx]:.0f}")

    ### Update plot with final image and markers

    # Final image update
    kpl.imshow_update(ax, img_array)

    # Mark optimal position
    if opti_x is not None and opti_y is not None:
        ax.plot(opti_x, opti_y, "g*", markersize=20, zorder=10,
                label=f"Optimal ({opti_x:.4f}, {opti_y:.4f})")

        # Draw circle showing fit sigma if available
        if fit_params is not None:
            sigma = fit_params["sigma"]
            circle = plt.Circle((opti_x, opti_y), sigma, fill=False,
                               color="lime", linestyle="--", linewidth=2,
                               label=f"1σ = {sigma:.4f}", zorder=9,
                               clip_on=True)  # Clip to axes bounds
            ax.add_patch(circle)

    # Re-mark initial position
    ax.plot(center_x, center_y, "r+", markersize=15, markeredgewidth=2, zorder=10,
            label="Initial position")

    ax.legend(loc="upper right", facecolor='white', framealpha=0.8)

    # Set explicit axis limits to prevent circle from expanding plot area
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    plt.pause(0.1)

    ### Move to optimal position

    opti_counts = None
    if opti_x is not None and opti_y is not None and move_to_optimal:
        print(f"\nMoving to optimal position...")
        pos.set_xyz((opti_x, opti_y), positioner=CoordsKey.PIXEL)
        time.sleep(0.05)  # Settling time

        # Measure counts at optimal position
        counter.start_tag_stream()
        samples = []
        for _ in range(5):  # Average 5 samples for verification
            pulse_gen.stream_start(1)
            raw = counter.read_counter_simple(1)
            if raw:
                samples.append(int(raw[0]))
        counter.stop_tag_stream()

        opti_counts = int(np.mean(samples)) if samples else 0
        print(f"  Counts at optimal position: {opti_counts}")

    plt.ioff()
    tb.reset_cfm()

    ### Prepare results

    results = {
        "opti_x": float(opti_x) if opti_x is not None else None,
        "opti_y": float(opti_y) if opti_y is not None else None,
        "opti_counts": opti_counts,
        "img_array": img_array.tolist(),
        "x_vals": x_vals.tolist(),
        "y_vals": y_vals.tolist(),
        "fit_params": fit_params,
        "fit_success": fit_success,
        "initial_coords": initial_coords,
        "img_extent": extent,
        "scan_params": {
            "num_steps": num_steps,
            "optimize_range": optimize_range,
            "fit_method": fit_method,
            "num_averages": num_averages,
        },
    }

    ### Save data

    if save_data:
        timestamp = dm.get_time_stamp()
        raw_data = {
            "timestamp": timestamp,
            "nv_sig": nv_sig,
            "optimization_results": results,
        }
        nv_name = getattr(nv_sig, "name", "unknown")
        file_path = dm.get_file_path(__file__, timestamp, f"{nv_name}_xy_optimize")
        dm.save_raw_data(raw_data, file_path)
        dm.save_figure(fig, file_path)
        print(f"\nData saved to: {file_path}")

    print(f"\n{'='*50}")
    print("XY OPTIMIZATION COMPLETE")
    if opti_x is not None and opti_y is not None:
        print(f"  Optimal position: X={opti_x:.4f}, Y={opti_y:.4f}")
        if opti_counts is not None:
            print(f"  Counts at optimal: {opti_counts}")
    print(f"{'='*50}\n")

    kpl.show()

    return results


if __name__ == "__main__":
    """Example usage for testing"""
    from utils.constants import NVSig, CoordsKey, VirtualLaserKey

    # Create a minimal nv_sig for testing
    nv_sig = NVSig(
        name="test_xy_optimize",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.PIXEL: [0.0, 0.0], CoordsKey.Z: 0},
        pulse_durations={VirtualLaserKey.IMAGING: int(1e6)},  # 1 ms
    )

    # Run optimization
    results = main(
        nv_sig,
        num_steps=15,  # 15x15 = 225 points
        fit_method="gaussian",  # or "max_counts"
    )
