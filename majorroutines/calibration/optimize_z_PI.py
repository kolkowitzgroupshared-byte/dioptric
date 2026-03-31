# -*- coding: utf-8 -*-
"""
Z-axis optimization routine for PI E-709 piezo using voltage scan and Gaussian fit.

Scans the piezo through a voltage range, collects photon counts at each position,
fits a 1D Gaussian to find the optimal Z position, and moves to the peak.

Uses GCS qPOS() command to read actual piezo position for accurate data logging.

Created by: chemistatcode
Created on March 5th, 2026


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


def gaussian_1d(x, amplitude, center, sigma, offset):
    """
    1D Gaussian function for fitting.

    Parameters
    ----------
    x : np.ndarray
        Position values
    amplitude : float
        Peak height above offset
    center : float
        Center position
    sigma : float
        Standard deviation
    offset : float
        Background offset

    Returns
    -------
    np.ndarray
        Gaussian values at each position
    """
    return offset + amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def optimize_z_PI(
    nv_sig: NVSig,
    voltage_start: float,
    voltage_end: float,
    step_size: float = 0.01,
    num_averages: int = 3,
    move_to_optimal: bool = True,
    save_data: bool = True,
    use_position_feedback: bool = False,
) -> dict:
    """
    Optimize Z position for PI E-709 piezo using voltage scan and Gaussian fit.

    This function:
    1. Scans through the specified voltage range
    2. Collects photon counts at each position
    3. Reads actual position via qPOS() for accurate logging
    4. Fits a 1D Gaussian to find optimal Z
    5. Optionally moves to optimal position

    Parameters
    ----------
    nv_sig : NVSig
        NV center parameters (pulse durations, laser settings)
    voltage_start : float
        Starting voltage (V). Must be in range [1.0, 9.0].
    voltage_end : float
        Ending voltage (V). Must be in range [1.0, 9.0].
    step_size : float, optional
        Voltage step size (V). Default: 0.01V (10mV)
    num_averages : int, optional
        Number of photon count samples to average at each position. Default: 3
    move_to_optimal : bool, optional
        Whether to move piezo to optimal position after fitting. Default: True
    save_data : bool, optional
        Whether to save data and plot. Default: True
    use_position_feedback : bool, optional
        Whether to use qPOS() to read actual position. Default: False.
        Note: qPOS() may timeout in external control mode. When False,
        position is estimated from voltage using ~6.0 µm/V conversion.

    Returns
    -------
    dict
        Results containing:
        - opti_voltage: Optimal voltage (V)
        - opti_position_um: Optimal position in micrometers
        - opti_counts: Counts at optimal position
        - voltages: Array of scanned voltages
        - positions_um: Array of actual positions from qPOS()
        - counts: Array of photon counts
        - fit_params: Gaussian fit parameters (if successful)
        - fit_success: Whether Gaussian fit succeeded
        - initial_voltage: Starting voltage before optimization
    """

    ### Validate inputs
    if not (1.0 <= voltage_start <= 9.0):
        raise ValueError(f"voltage_start must be in range [1.0, 9.0], got {voltage_start}")
    if not (1.0 <= voltage_end <= 9.0):
        raise ValueError(f"voltage_end must be in range [1.0, 9.0], got {voltage_end}")
    if voltage_start >= voltage_end:
        raise ValueError(f"voltage_start ({voltage_start}) must be less than voltage_end ({voltage_end})")
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")

    ### Setup
    config = common.get_config_dict()

    # Get hardware servers
    piezo_server = pos.get_positioner_server(CoordsKey.Z)
    counter = tb.get_server_counter()
    pulse_gen = tb.get_server_pulse_streamer()

    # Calculate voltage array
    voltages = np.arange(voltage_start, voltage_end + step_size / 2, step_size)
    num_steps = len(voltages)

    # Get current voltage for reference
    initial_voltage = piezo_server.read_z()

    # Setup laser for imaging
    laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.IMAGING)
    readout_ns = int(
        nv_sig.pulse_durations.get(VirtualLaserKey.IMAGING, int(laser_dict["duration"]))
    )
    laser_name = laser_dict["physical_name"]

    print(f"\nPI E-709 Z Optimization")
    print(f"=" * 50)
    print(f"Initial voltage: {initial_voltage:.3f} V")
    print(f"Scan range: {voltage_start:.3f} V to {voltage_end:.3f} V")
    print(f"Step size: {step_size:.4f} V ({step_size * 1000:.1f} mV)")
    print(f"Number of steps: {num_steps}")
    print(f"Averages per point: {num_averages}")
    print(f"=" * 50 + "\n")

    ### Setup figure for real-time display
    kpl.init_kplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    if use_position_feedback:
        ax.set_xlabel("Position (µm)")
    else:
        ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Counts")
    ax.set_title(f"PI E-709 Z Optimization - {laser_name}, {readout_ns/1e6:.1f} ms")
    ax.grid(True, alpha=0.3)
    (line,) = ax.plot([], [], "b.-", markersize=8, linewidth=1)
    plt.ion()
    plt.pause(0.1)

    ### Hardware setup
    tb.reset_cfm()
    tb.init_safe_stop()
    counter.start_tag_stream()

    # Load pulse sequence
    seq_file = "simple_readout.py"
    positioner_dict = config["Positioning"]["Positioners"][CoordsKey.Z]
    delay_ns = int(positioner_dict.get("delay", 0))

    seq_args = [delay_ns, readout_ns, laser_name, 1.0]
    pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args))

    ### Data collection
    positions_um = []
    counts_list = []
    # use_position_feedback is passed as parameter (default False due to external control mode)

    try:
        print(f"Scanning {num_steps} positions...")

        for i, voltage in enumerate(voltages):
            if tb.safe_stop():
                print("\n[STOPPED] User interrupt")
                break

            # Move to voltage
            piezo_server.write_z(voltage)
            time.sleep(0.02)  # 20ms settling time

            # Read actual position via qPOS()
            if use_position_feedback:
                try:
                    position_um = piezo_server.get_z_position()
                except Exception as e:
                    # Fallback: estimate from voltage (~6.0 µm/V empirically)
                    position_um = (voltage - 1.0) * 6.0
                    if i == 0:
                        print(f"[Warning] qPOS() failed: {e}")
                        print("Falling back to voltage-based position estimate")
                        use_position_feedback = False
            else:
                position_um = (voltage - 1.0) * 6.0

            positions_um.append(position_um)

            # Collect photon counts
            samples = []
            for _ in range(num_averages):
                pulse_gen.stream_start(1)
                raw = counter.read_counter_simple(1)
                if raw and len(raw) > 0:
                    samples.append(int(raw[0]))

            avg_counts = np.mean(samples) if samples else 0
            counts_list.append(avg_counts)

            # Update plot (use voltage for x-axis if no position feedback)
            if use_position_feedback:
                line.set_data(positions_um, counts_list)
            else:
                line.set_data(voltages[:i+1], counts_list)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

            # Progress output
            if (i + 1) % 5 == 0 or i == 0:
                print(
                    f"  Step {i+1}/{num_steps}: V={voltage:.3f}, "
                    f"Pos={position_um:.2f} µm, Counts={avg_counts:.0f}"
                )

    finally:
        counter.clear_buffer()
        tb.reset_cfm()
        tb.reset_safe_stop()

    print(f"\nCollected {len(counts_list)} data points")

    ### Convert to arrays
    positions = np.array(positions_um)
    counts = np.array(counts_list)
    voltages_scanned = voltages[: len(counts)]  # In case of early stop

    ### Gaussian fit
    # When position feedback is off, fit voltage directly; otherwise fit position
    if use_position_feedback:
        x_data = positions
        x_label = "µm"
    else:
        x_data = voltages_scanned
        x_label = "V"

    opti_voltage = None
    opti_position = None
    fit_params = None
    fit_success = False

    if len(counts) >= 5:
        try:
            # Initial guesses from data
            max_idx = np.argmax(counts)
            offset_guess = np.min(counts)
            amplitude_guess = np.max(counts) - offset_guess
            center_guess = x_data[max_idx]
            sigma_guess = (x_data[-1] - x_data[0]) / 4

            guess = [amplitude_guess, center_guess, sigma_guess, offset_guess]
            bounds = (
                [0, x_data[0], 0, 0],
                [np.inf, x_data[-1], np.inf, np.inf],
            )

            popt, _ = curve_fit(
                gaussian_1d, x_data, counts, p0=guess, bounds=bounds, maxfev=10000
            )

            opti_center = popt[1]
            fit_params = {
                "amplitude": float(popt[0]),
                "center": float(popt[1]),
                "sigma": float(popt[2]),
                "offset": float(popt[3]),
            }
            fit_success = True

            if use_position_feedback:
                opti_position = opti_center
                # Convert position back to voltage (estimated)
                opti_voltage = 1.0 + (opti_position / 6.0)
                opti_voltage = max(1.0, min(9.0, opti_voltage))
            else:
                opti_voltage = opti_center
                opti_voltage = max(1.0, min(9.0, opti_voltage))
                # Estimate position from voltage
                opti_position = (opti_voltage - 1.0) * 6.0

            # Plot fit curve
            x_fit = np.linspace(x_data[0], x_data[-1], 200)
            counts_fit = gaussian_1d(x_fit, *popt)
            ax.plot(x_fit, counts_fit, "r-", linewidth=2, label="Gaussian fit")
            ax.axvline(
                opti_center,
                color="g",
                linestyle="--",
                linewidth=2,
                label=f"Optimal: {opti_center:.4f} {x_label}",
            )
            ax.legend(loc="upper right")

            print(f"\nGaussian fit successful:")
            print(f"  Optimal voltage: {opti_voltage:.4f} V")
            if use_position_feedback:
                print(f"  Optimal position: {opti_position:.2f} µm")
            print(f"  Amplitude: {fit_params['amplitude']:.0f} counts")
            print(f"  Sigma: {fit_params['sigma']:.4f} {x_label}")
            print(f"  Offset: {fit_params['offset']:.0f} counts")

        except Exception as e:
            print(f"\nGaussian fit failed: {e}")
            print("Using max counts position")
            max_idx = np.argmax(counts)
            opti_voltage = voltages_scanned[max_idx]
            opti_position = positions[max_idx]
            print(f"  Max counts at: {opti_voltage:.4f} V")

    else:
        print("\nInsufficient data for fitting, using max counts")
        if len(counts) > 0:
            max_idx = np.argmax(counts)
            opti_voltage = voltages_scanned[max_idx]
            opti_position = positions[max_idx]

    plt.pause(0.1)

    ### Move to optimal position
    opti_counts = None
    if opti_voltage is not None and move_to_optimal:
        print(f"\nMoving to optimal voltage: {opti_voltage:.4f} V")
        piezo_server.write_z(opti_voltage)
        time.sleep(0.05)  # Settling time

        # Verify counts at optimal position
        counter.start_tag_stream()
        samples = []
        for _ in range(5):
            pulse_gen.stream_start(1)
            raw = counter.read_counter_simple(1)
            if raw:
                samples.append(int(raw[0]))
        counter.stop_tag_stream()

        opti_counts = int(np.mean(samples)) if samples else 0
        print(f"  Counts at optimal: {opti_counts}")

        # Mark on plot
        if opti_voltage is not None:
            plot_x = opti_position if use_position_feedback else opti_voltage
            ax.plot(
                plot_x,
                opti_counts,
                "g*",
                markersize=20,
                zorder=10,
                label=f"Final: {opti_counts} counts",
            )
            ax.legend(loc="upper right")

    plt.ioff()
    tb.reset_cfm()

    ### Prepare results
    results = {
        "opti_voltage": float(opti_voltage) if opti_voltage is not None else None,
        "opti_position_um": float(opti_position) if opti_position is not None else None,
        "opti_counts": opti_counts,
        "voltages": voltages_scanned.tolist(),
        "positions_um": positions.tolist(),
        "counts": counts.tolist(),
        "fit_params": fit_params,
        "fit_success": fit_success,
        "initial_voltage": float(initial_voltage),
        "use_position_feedback": use_position_feedback,
        "scan_params": {
            "voltage_start": voltage_start,
            "voltage_end": voltage_end,
            "step_size": step_size,
            "num_steps": num_steps,
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
        file_path = dm.get_file_path(__file__, timestamp, f"{nv_name}_z_optimize_PI")
        dm.save_raw_data(raw_data, file_path)
        dm.save_figure(fig, file_path)
        print(f"\nData saved to: {file_path}")

    print(f"\n{'=' * 50}")
    print("Z OPTIMIZATION COMPLETE")
    if opti_voltage is not None:
        print(f"  Optimal voltage: {opti_voltage:.4f} V")
    if opti_position is not None:
        print(f"  Optimal position: {opti_position:.2f} µm")
    if opti_counts is not None:
        print(f"  Counts at optimal: {opti_counts}")
    print(f"{'=' * 50}\n")

    kpl.show()

    return results


if __name__ == "__main__":
    """Example usage for testing"""
    from utils.constants import CoordsKey, NVSig, VirtualLaserKey

    # Create a minimal nv_sig for testing
    nv_sig = NVSig(
        name="test_z_optimize_PI",
        coords={CoordsKey.SAMPLE: [0.0, 0.0], CoordsKey.PIXEL: [0.0, 0.0], CoordsKey.Z: 4.0},
        pulse_durations={VirtualLaserKey.IMAGING: int(10e6)},  # 10 ms
    )

    # Run optimization - example with 10mV steps over 0.2V range
    results = optimize_z_PI(
        nv_sig,
        voltage_start=3.90,
        voltage_end=4.10,
        step_size=0.01,
        num_averages=3,
    )
