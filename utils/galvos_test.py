# import threading
# import time

# import nidaqmx
# import numpy as np
# from nidaqmx.constants import TerminalConfiguration


# # Write a single voltage value to X galvo (ao22)
# def write_x(voltage):
#     with nidaqmx.Task() as task:
#         task.ao_channels.add_ao_voltage_chan("Dev1/ao22")
#         task.write(voltage)


# # Write a single voltage value to Y galvo (ao31)
# def write_y(voltage):
#     with nidaqmx.Task() as task:
#         task.ao_channels.add_ao_voltage_chan("Dev1/ao31")
#         task.write(voltage)


# # Run this in a thread so it can be stopped externally
# def run_pattern(stop_event, freq_x=0.5, freq_y=0.4, amp_x=1.0, amp_y=1.0, rate=1000):
#     task = nidaqmx.Task()
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao22")  # X
#     task.ao_channels.add_ao_voltage_chan("Dev1/ao31")  # Y

#     t = 0
#     dt = 1 / rate

#     try:
#         while not stop_event.is_set():
#             vx = amp_x * np.sin(2 * np.pi * freq_x * t)
#             vy = amp_y * np.sin(2 * np.pi * freq_y * t)
#             task.write([vx, vy])
#             time.sleep(dt)
#             t += dt
#     finally:
#         task.close()


# stop_event = threading.Event()
# try:
#     thread = threading.Thread(target=run_pattern, args=(stop_event,))
#     thread.start()
#     while thread.is_alive():
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     print("Stopping...")
#     stop_event.set()
#     thread.join()

# # Start pattern in background

# thread.start()

# # Let it run for 10 seconds
# time.sleep(10)

# # Stop
# stop_event.set()
# thread.join()
import threading
import time

import nidaqmx
import numpy as np
from utils import common

# # === Configurable parameters ===
# DEVICE_NAME = "Dev1"
# X_CHANNEL = "ao22"
# Y_CHANNEL = "ao31"

# x_min = -4.0  # Safe voltage range
# x_max = 4.0
# y_min = -4.0
# y_max = 4.0

# x_points = 100  # Resolution along X
# y_points = 100  # Resolution along Y
# rate = 1000  # Point update rate (Hz)
# return_delay = 0.001  # Delay at end of each line
                                               

# # === Raster scan function ===
# def raster_scan(stop_event):
#     task = nidaqmx.Task()
#     task.ao_channels.add_ao_voltage_chan(f"{DEVICE_NAME}/{X_CHANNEL}")
#     task.ao_channels.add_ao_voltage_chan(f"{DEVICE_NAME}/{Y_CHANNEL}")

#     try:
#         x_vals = np.linspace(x_min, x_max, x_points)
#         y_vals = np.linspace(y_min, y_max, y_points)

#         while not stop_event.is_set():
#             for y in y_vals:
#                 if stop_event.is_set():
#                     break
#                 for x in x_vals:
#                     task.write([x, y])  # X changes fast, Y is constant per line
#                     time.sleep(1 / rate)
#                 time.sleep(return_delay)  # Give galvo time to return
#     finally:
#         print("Stopping scan and parking galvos at center.")
#         task.write([0.0, 0.0])  # Park galvos at center
#         task.close()


# # Start raster scan 
# stop_event = threading.Event()
# thread = threading.Thread(target=raster_scan, args=(stop_event,))
# thread.start()

# # Let it run for 10 seconds
# time.sleep(60)

# # Stop the scan
# stop_event.set()
# thread.join()


# === Configurable parameters ===
#DEVICE_NAME = "Dev1"
#X_CHANNEL = "ao11"
#Y_CHANNEL = "ao4"

cxn = common.labrad_connect()
opx = cxn.pos_xy_THOR_gvs212
amp_x = 0.4  # amplitude (safe range)
amp_y = 0.4 
freq_x = 4  # Hz
freq_y = 6
rate = 1000  # update rate (Hz)


# === Continuous Lissajous pattern ===
def lissajous_scan(stop_event):
    #task = nidaqmx.Task()
    #task.ao_channels.add_ao_voltage_chan(f"{DEVICE_NAME}/{X_CHANNEL}")
    #task.ao_channels.add_ao_voltage_chan(f"{DEVICE_NAME}/{Y_CHANNEL}")

    t = 0
    dt = 1 / rate

    try:
        while not stop_event.is_set():
            x = amp_x * np.sin(2 * np.pi * freq_x * t)
            y = amp_y * np.sin(2 * np.pi * freq_y * t)
            opx.write_xy(x, y)
            time.sleep(dt)
            t += dt
    finally:
        print("Stopping scan and parking galvos at center.")
        opx.write_xy(0.0, 0.0)  # Park galvos at center


# Start Lissajous pattern
stop_event = threading.Event()
thread = threading.Thread(target=lissajous_scan, args=(stop_event,))
thread.start()

# Let it run for 10 seconds (or remove this to run indefinitely)
time.sleep(10)

# # Stop
stop_event.set()
thread.join()
