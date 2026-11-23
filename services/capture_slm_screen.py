"""Capture what Windows is currently showing on the SLM's display, save it as
slm_screen.png and show it.

The SLM is driven like an extra monitor (display 2, 1920 x 1080, right of the main
screen). A loaded hologram looks like grey, noise-like texture; a blank SLM is one
flat grey level; a piece of desktop or another program's window means something
else is covering the pattern.

Run in the dioptric environment, while the SLM script is holding its pattern:
    python capture_slm_screen.py
"""

import ctypes
from ctypes import wintypes

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageGrab

user32 = ctypes.windll.user32
try:
    user32.SetProcessDPIAware()  # work in real pixels, as Windows reports the displays
except Exception:
    pass


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
displays = []  # (name, left, top, right, bottom, primary)


def add_display(hmonitor, hdc, rect, lparam):
    info = MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
    user32.GetMonitorInfoW(hmonitor, ctypes.byref(info))
    r = info.rcMonitor
    displays.append((info.szDevice, r.left, r.top, r.right, r.bottom, bool(info.dwFlags & 1)))
    return 1


callback = MonitorEnumProc(add_display)
user32.EnumDisplayMonitors(None, None, callback, 0)

# The SLM: the display that is not the main one, preferably 1920 x 1080
others = [d for d in displays if not d[5]]
if not others:
    raise SystemExit("Only the main display is connected: Windows does not see the SLM.")
sized = [d for d in others if (d[3] - d[1], d[4] - d[2]) == (1920, 1080)]
name, left, top, right, bottom, _ = (sized or others)[0]

# Screenshot of all displays; its origin is the top-left corner of the whole desktop
full = ImageGrab.grab(all_screens=True)
origin_x = min(d[1] for d in displays)
origin_y = min(d[2] for d in displays)
slm_image = full.crop((left - origin_x, top - origin_y, right - origin_x, bottom - origin_y))
slm_image.save("slm_screen.png")

gray = np.asarray(slm_image.convert("L"), dtype=float)
print(f"SLM display {name}: {right - left} x {bottom - top}, saved as slm_screen.png")
print(f"Grey levels: mean {gray.mean():.1f}, spread (std) {gray.std():.1f}, "
      f"from {gray.min():.0f} to {gray.max():.0f}")
print("A hologram has a large spread; a blank SLM has a spread near 0.")

plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title(f"What Windows shows on {name} (the SLM)")
plt.show()
