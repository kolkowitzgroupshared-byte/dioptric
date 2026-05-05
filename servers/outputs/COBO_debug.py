"""
Debugging laser_COBO_520 connection issues for power measurments Cryo

Created: 4/27/2026
@author: chemistatcode
"""
## Connection issue debug
# import serial
# s=serial.Serial("COM4",115200,timeout=1)
# print(s.is_open)
# s.close()

## Power changing debug
# import labrad
# cxn = labrad.connect()
# laser = cxn.laser_COBO_520

#   # What does the CW setpoint think it is, and does it move?
# print("p? before:", laser.get_power())
# laser.set_power(0.005)            # 5 mW
# print("p? after 5mW set:", laser.get_power())
# laser.set_power(0.010)            # 10 mW
# print("p? after 10mW set:", laser.get_power())
# print("pa?:", laser.get_actual_power())


# from utils import common
# cxn = common.labrad_connect()
# laser = cxn.laser_COBO_520

# for target_w in [0.001, 0.003, 0.005, 0.008, 0.010, 0.020, 0.050, 0.100, 0.005]:
#     laser.set_power(target_w)
#     readback = laser.get_power()
#     print(f"set {target_w*1e3:7.2f} mW  ->  p? = {readback*1e3:7.2f} mW   "
#         f"(match: {abs(readback - target_w) < 1e-6})")

# from utils import common
# cxn = common.labrad_connect()
# laser = cxn.laser_COBO_520

# print("== p (CW setpoint) ==")
# for target_w in [0.001, 0.003, 0.005, 0.008, 0.010, 0.005]:
#     laser.set_power(target_w)
#     rb = laser.get_power()
#     print(f"  set {target_w*1e3:7.2f} mW  ->  p?     = {rb*1e3:7.2f} mW   "
#         f"({'OK' if abs(rb-target_w)<1e-6 else 'MISMATCH'})")

# print("== slmp (modulation setpoint) ==")
# for target_w in [0.001, 0.003, 0.005, 0.008, 0.010, 0.005]:
#     laser.set_modulation_power(target_w)
#     rb = laser.get_modulation_power()
#     print(f"  set {target_w*1e3:7.2f} mW  ->  glmp?  = {rb*1e3:7.2f} mW   "
#         f"({'OK' if abs(rb-target_w)<1e-6 else 'MISMATCH'})")

## Unit confirmation
from utils import common
cxn = common.labrad_connect()
laser = cxn.laser_COBO_520

print("glmp? right now:", laser.get_modulation_power())

try:
    laser.set_modulation_power(1.0)
    print("slmp 1.0 accepted; glmp? =", laser.get_modulation_power())
    print("=> unit is most likely mW")
except Exception as e:
    print("slmp 1.0 rejected:", e)
    print("=> unit is W (Cobolt refused 1 W as out of range)")