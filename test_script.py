from utils import common

p = common.labrad_connect().picomotor_8742
print("IDN:", p.get_idn())
print("xyz:", p.get_xyz())
# print(p.move_rel(1, 100))
# print(p.move_rel(1, -100))
