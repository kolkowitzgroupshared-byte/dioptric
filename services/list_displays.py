"""List every display Windows currently sees, with its size and position.

The SLM is driven like an extra monitor, so if it is connected and switched on it
shows up here as an additional 1920 x 1080 display, next to your normal monitors.

Run in the dioptric environment:
    python list_displays.py
"""

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
try:
    user32.SetProcessDPIAware()  # report real pixel sizes, not scaled ones
except Exception:
    pass

MONITORINFOF_PRIMARY = 1


class MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


MonitorEnumProc = ctypes.WINFUNCTYPE(
    ctypes.c_int, wintypes.HANDLE, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
)

displays = []


def add_display(hmonitor, hdc, rect, lparam):
    info = MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
    user32.GetMonitorInfoW(hmonitor, ctypes.byref(info))
    r = info.rcMonitor
    primary = bool(info.dwFlags & MONITORINFOF_PRIMARY)
    displays.append((info.szDevice, r.right - r.left, r.bottom - r.top, r.left, r.top, primary))
    return 1


callback = MonitorEnumProc(add_display)  # keep a reference while Windows calls it
user32.EnumDisplayMonitors(None, None, callback, 0)

for ind, (name, width, height, x, y, primary) in enumerate(displays):
    main = "  (main display)" if primary else ""
    print(f"{ind}: {name}  {width} x {height}  at ({x}, {y}){main}")
print(f"{len(displays)} display(s) found. The SLM would be an extra 1920 x 1080 display.")
