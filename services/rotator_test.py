# -*- coding: utf-8 -*-
"""
Created on Nov 5, 2025

@author: Alyssa Matthews

"""

import time

from utils import common
from utils import tool_belt as tb

rotation_server = tb.get_server_rotation_mount()
rotation_server.set_angle(330)

time.sleep(1)

angle = rotation_server.get_angle()
print("Angle:", angle, "degrees")
