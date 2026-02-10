# -*- coding: utf-8 -*-
"""
Created on October 27th, 2025

@author: Eric Gediman, Alyssa Matthews

"""

from utils import common
from utils import tool_belt as tb

# print(tb.dummyfunc())
power_meter_server = tb.get_server_power_meter()
response = power_meter_server.set_wavelength(800)
print("Response: ", response)
print("Wavelength: ", power_meter_server.get_wavelength())
print("Power: ", power_meter_server.get_power())

# cxn = common.labrad_connect()
# opx = cxn.power_meter_THOR_pm100d
# print("Power: ", opx.get_power())
