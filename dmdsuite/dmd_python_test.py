import ctypes
from pathlib import Path
from PIL import Image
import time

DLL_PATH = r"C:\Users\jkdol\OneDrive\Documents\Github\dioptric\dmdsuite\Windows_x86_64\DLL_x64\x64\Release\DLP6500_DLL.dll"
IMAGE_PATH = r"C:\Users\jkdol\OneDrive\Documents\Github\dioptric\dmdsuite\Windows_x86_64\opticaltweezers.jpg"

DMD_W = 1920
DMD_H = 1080
DMD_N = DMD_W * DMD_H

PLANE_OFF = 200
PLANE_SPOT = 201

dll_path = Path(DLL_PATH)
if not dll_path.exists():
    raise FileNotFoundError(f"DLL not found: {dll_path}")

DlpDLL = ctypes.WinDLL(str(dll_path))

GetDevice_Proto = ctypes.WINFUNCTYPE(ctypes.c_void_p)
GetDevice = GetDevice_Proto(("GetDevice", DlpDLL), ())

Connect_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
Connect = Connect_Proto(("Connect", DlpDLL), ((1, "hdev"), (1, "dev_id")))

Disconnect_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
Disconnect = Disconnect_Proto(("Disconnect", DlpDLL), ((1, "hdev"),))

DeleteDevice_Proto = ctypes.WINFUNCTYPE(None, ctypes.c_void_p)
DeleteDevice = DeleteDevice_Proto(("DeleteDevice", DlpDLL), ((1, "hdev"),))

StopSequence_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
StopSequence = StopSequence_Proto(("StopSequence", DlpDLL), ((1, "hdev"),))

RunSequence_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
RunSequence = RunSequence_Proto(("RunSequence", DlpDLL), ((1, "hdev"), (1, "startpos")))

SendImageMono_Proto = ctypes.WINFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_ubyte * DMD_N
)
SendImageMono_c = SendImageMono_Proto(
    ("SendImageMono", DlpDLL),
    ((1, "hdev"), (1, "planenr"), (1, "buffer"))
)

ListControllers_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
ListControllers = ListControllers_Proto(("ListControllers", DlpDLL), ((1, "count"),))

GetDevID_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
GetDevID = GetDevID_Proto(("GetDevID", DlpDLL), ((1, "index"), (1, "dev_id")))

IsConnected_Proto = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool))
IsConnected = IsConnected_Proto(("IsConnected", DlpDLL), ((1, "hdev"), (1, "connected")))


def make_off_plane():
    return Image.new("L", (DMD_W, DMD_H), 0)


def make_canvas_from_image(path, threshold=128):
    im = Image.open(path).convert("L")
    print("original size:", im.size, flush=True)
    print("original mode:", im.mode, flush=True)

    if im.size[0] > DMD_W or im.size[1] > DMD_H:
        im.thumbnail((DMD_W, DMD_H))

    canvas = Image.new("L", (DMD_W, DMD_H), 0)
    x0 = (DMD_W - im.size[0]) // 2
    y0 = (DMD_H - im.size[1]) // 2
    canvas.paste(im, (x0, y0))

    if threshold is not None:
        canvas = canvas.point(lambda p: 255 if p >= threshold else 0)

    return canvas


def send_image_mono(hdev, planenr, pil_image):
    if pil_image.mode != "L":
        raise ValueError(f"Image mode must be L, got {pil_image.mode}")
    if pil_image.size != (DMD_W, DMD_H):
        raise ValueError(f"Image size must be {(DMD_W, DMD_H)}, got {pil_image.size}")

    greyvals = bytearray(pil_image.tobytes())
    if len(greyvals) != DMD_N:
        raise ValueError(f"Expected {DMD_N} bytes, got {len(greyvals)}")

    buf = (ctypes.c_ubyte * DMD_N).from_buffer(greyvals)
    return SendImageMono_c(hdev, planenr, buf)


def require_zero(name, rc):
    print(f"{name}: {rc}", flush=True)
    if rc != 0:
        raise RuntimeError(f"{name} failed with rc={rc}")


def main():
    print("creating handle...", flush=True)
    hdev = GetDevice()
    print("hdev =", hdev, flush=True)

    count = ctypes.c_uint(0)
    require_zero("ListControllers", ListControllers(ctypes.byref(count)))
    print("controller count =", count.value, flush=True)
    if count.value < 1:
        raise RuntimeError("No DMD controllers found")

    dev_id = ctypes.c_int(-1)
    require_zero("GetDevID", GetDevID(0, ctypes.byref(dev_id)))
    print("dev_id =", dev_id.value, flush=True)

    time.sleep(0.2)

    require_zero("Connect", Connect(hdev, dev_id.value))

    connected = ctypes.c_bool(False)
    require_zero("IsConnected", IsConnected(hdev, ctypes.byref(connected)))
    print("connected =", connected.value, flush=True)
    if not connected.value:
        raise RuntimeError("Device reports not connected")

    off = make_off_plane()
    spot = make_canvas_from_image(IMAGE_PATH, threshold=128)
    print("off size:", off.size, "spot size:", spot.size, flush=True)

    require_zero("SendImageMono off", send_image_mono(hdev, PLANE_OFF, off))
    require_zero("SendImageMono spot", send_image_mono(hdev, PLANE_SPOT, spot))

    require_zero("StopSequence", StopSequence(hdev))
    require_zero("RunSequence off", RunSequence(hdev, PLANE_OFF))
    input("Check OFF plane, then press Enter...")

    require_zero("StopSequence", StopSequence(hdev))
    require_zero("RunSequence spot", RunSequence(hdev, PLANE_SPOT))
    input("Check SPOT plane, then press Enter...")

    require_zero("StopSequence", StopSequence(hdev))
    require_zero("RunSequence off again", RunSequence(hdev, PLANE_OFF))
    input("Check OFF again, then press Enter to exit...")


    # show OFF for 1 second
    require_zero("StopSequence", StopSequence(hdev))
    require_zero("RunSequence off", RunSequence(hdev, PLANE_OFF))
    time.sleep(1.0)

    # show SPOT for 1 second
    require_zero("StopSequence", StopSequence(hdev))
    require_zero("RunSequence spot", RunSequence(hdev, PLANE_SPOT))
    time.sleep(1.0)

    # back to OFF
    require_zero("StopSequence", StopSequence(hdev))
    require_zero("RunSequence off again", RunSequence(hdev, PLANE_OFF))
    time.sleep(1.0)

    print("disconnect:", Disconnect(hdev), flush=True)
    DeleteDevice(hdev)
    print("done", flush=True)


if __name__ == "__main__":
    main()
