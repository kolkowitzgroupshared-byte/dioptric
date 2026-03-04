from utils import common
from utils import tool_belt as tb

cxn = common.labrad_connect()
opx = cxn.multimeter_KEIT_daq6510

print(opx.read_temperature(101))

