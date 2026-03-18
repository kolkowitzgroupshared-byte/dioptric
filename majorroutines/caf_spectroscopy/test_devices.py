"""
Search for NV triplet-to-singlet wavelength

Created on November 3, 2025

@author: jchen, Alyssa Matthews
"""

import time

from utils import common
from utils import tool_belt as tb


def main():
    """
    Main function to run the power measurement sequence.
    """

    power_meter_server = tb.get_server_power_meter()  # use get_power() command
    slider_2_server = tb.get_server_slider_2()  # use set_filter(n) command
    # n is the slot you want to move the slider to
    # (3 = power msmt, 2 = beam passes thru)

    try:
        # 1. Move slider to position 3
        slider_2_server.set_filter(3)

        # 2. Pause and read power
        pause_duration = 0.5
        time.sleep(pause_duration)
        print("Power: ", power_meter_server.get_power(), "W")
        time.sleep(pause_duration)

        # 3. Move slider to position 2
        slider_2_server.set_filter(2)

    except Exception as e:
        print(f"An error occurred during the sequence: {e}")


if __name__ == "__main__":
    main()
