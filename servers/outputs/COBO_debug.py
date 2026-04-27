"""
Debugging laser_COBO_520 connection issues for power measurments Cryo

Created: 4/27/2026
@author: chemistatcode
"""

import serial
s=serial.Serial("COM4",115200,timeout=1)
print(s.is_open)
s.close()