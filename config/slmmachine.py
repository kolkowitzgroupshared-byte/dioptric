# -*- coding: utf-8 -*-
"""
Config file for Thorslm server

Created June 21st, 2024

@author: sbchand
"""
from pathlib import Path

# Set home directory path
home = Path.home()

# Configuration dictionary
config = {
    "shared_email": "kolkowitznvlab@gmail.com",
    "windows_repo_path": home / "Documents/dioptric",
    "linux_repo_path": home / "Documents/dioptric",
    "windows_nvdata_path": Path("G:/nvdata"),  # <- rename from _dir to _path
    "Servers": {
        "thorslm": "slm_THOR_exulus_hd2",
    },
}


def print_config(configuration):
    """
    Function to print the configuration dictionary.

    Parameters:
    configuration (dict): The configuration dictionary to print.
    """
    for key, value in configuration.items():
        print(f"{key}: {value}")


# Call the function to print configuration for verification
if __name__ == "__main__":
    print_config(config)
