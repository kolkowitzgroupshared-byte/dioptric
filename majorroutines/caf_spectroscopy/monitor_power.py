import collections
import time

import matplotlib.pyplot as plt

from utils import tool_belt as tb


def get_scale_factor(max_power):
    """Dynamically determines the best unit prefix based on the max power."""
    if max_power >= 1:
        return 1, "W"
    elif max_power >= 1e-3:
        return 1e3, "mW"
    elif max_power >= 1e-6:
        return 1e6, "µW"
    elif max_power >= 1e-9:
        return 1e9, "nW"
    else:
        return 1e12, "pW"


def main():
    """
    Live plotting of laser power for alignment.
    Dynamically auto-scales to mW, µW, nW, etc.
    """
    # 1. Connect to the server
    power_meter = tb.get_server_power_meter()
    if power_meter is None:
        print("Error: Could not connect to the power meter server.")
        return

    # 2. Configure the rolling window
    window_seconds = 15
    delay = 0.2  # 5 frames per second
    max_data_points = int(window_seconds / delay)

    # Use deque for times and raw power in Watts
    times = collections.deque(maxlen=max_data_points)
    raw_powers = collections.deque(maxlen=max_data_points)

    # 3. Set up the live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    (line,) = ax.plot([], [], lw=2, color="blue")

    ax.set_title("Live Laser Power Optimization")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle="--", alpha=0.7)

    print("=== Live Graph Active ===")
    print("Close the graph window or press Ctrl+C in the terminal to stop.")

    start_time = time.time()

    try:
        while True:
            current_time = time.time() - start_time

            # Fetch raw power, convert the PyVISA string to a decimal float, and append
            power_w = float(power_meter.get_power())
            times.append(current_time)
            raw_powers.append(power_w)

            # 4. DYNAMIC SCALING LOGIC
            # Find the max power currently on screen to determine the unit
            current_max_w = max(raw_powers)
            multiplier, unit = get_scale_factor(current_max_w)

            # Multiply all data points in the window by the scaling factor
            scaled_powers = [p * multiplier for p in raw_powers]

            # Update the line data with the scaled values
            line.set_data(times, scaled_powers)

            # Update the Y-axis label to show the current unit
            ax.set_ylabel(f"Power ({unit})")

            # Slide the X-axis
            ax.set_xlim(
                max(0, current_time - window_seconds), max(current_time, window_seconds)
            )

            # Dynamically scale the Y-axis using the newly scaled values
            min_p = min(scaled_powers)
            max_p = max(scaled_powers)
            buffer = (max_p - min_p) * 0.1 if max_p != min_p else 0.1
            ax.set_ylim(min_p - buffer, max_p + buffer)

            # Pause to update the graph
            plt.pause(delay)

    except KeyboardInterrupt:
        print("\nMonitoring stopped via terminal.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
