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


from utils import common
cxn = common.labrad_connect()
laser = cxn.laser_COBO_520

for target_w in [0.001, 0.003, 0.005, 0.008, 0.010, 0.020, 0.050, 0.100, 0.005]:
    laser.set_power(target_w)
    readback = laser.get_power()
    print(f"set {target_w*1e3:7.2f} mW  ->  p? = {readback*1e3:7.2f} mW   "
        f"(match: {abs(readback - target_w) < 1e-6})")