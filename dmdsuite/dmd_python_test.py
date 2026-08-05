# import sys
# import time
# import argparse
# from pathlib import Path
# import numpy as np
# from dmdsuite.dmd6500_api import Dmd6500, DMD_WIDTH, DMD_HEIGHT

# # Make repo importable
# repo = Path(__file__).resolve().parents[1]
# if str(repo) not in sys.path:
#     sys.path.append(str(repo))

# PASS_VALUE = 255   # ON / white = pass
# BLOCK_VALUE = 0    # OFF / black = block

# PASS_PLANE = 200
# BLOCK_PLANE = 201


# def all_mask(value):
#     return np.full((DMD_HEIGHT, DMD_WIDTH), value, dtype=np.uint8)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--method",
#         choices=["binary", "mono"],
#         default="binary",
#         help="binary uses SendPlane; mono uses SendImageMono",
#     )
#     args = parser.parse_args()

#     dll_path = (
#         repo
#         / "dmdsuite"
#         / "Windows_x86_64"
#         / "DLL_x64"
#         / "x64"
#         / "Release"
#         / "DLP6500_DLL.dll"
#     )

#     print("DLL path:", dll_path, flush=True)

#     dmd = None

#     try:
#         print("Creating Dmd6500 object...", flush=True)
#         dmd = Dmd6500(str(dll_path))
#         print("Dmd6500 object created.", flush=True)

#         print("Listing devices...", flush=True)
#         n = dmd.list_devices()
#         print("Devices found:", n, flush=True)

#         print("Connecting to DMD...", flush=True)
#         dmd.connect(0)
#         print("Connected.", flush=True)

#         print("Firmware version...", flush=True)
#         fw = dmd.firmware_version()
#         print("Firmware:", fw, flush=True)

#         # ------------------------------------------------------------
#         # PASS plane: white / ON
#         # ------------------------------------------------------------
#         pass_mask = all_mask(PASS_VALUE)

#         if args.method == "binary":
#             print("before send_binary_plane PASS", flush=True)
#             dmd.send_binary_plane(PASS_PLANE, pass_mask)
#             print("after send_binary_plane PASS", flush=True)
#         else:
#             print("before send_image_mono PASS", flush=True)
#             dmd.send_image_mono(PASS_PLANE, pass_mask)
#             print("after send_image_mono PASS", flush=True)

#         print("before show_plane PASS", flush=True)
#         dmd.show_plane(PASS_PLANE)
#         print("after show_plane PASS", flush=True)

#         print("PASS state should be visible now.")
#         time.sleep(2)

#         # ------------------------------------------------------------
#         # BLOCK plane: black / OFF
#         # ------------------------------------------------------------
#         block_mask = all_mask(BLOCK_VALUE)

#         if args.method == "binary":
#             print("before send_binary_plane BLOCK", flush=True)
#             dmd.send_binary_plane(BLOCK_PLANE, block_mask)
#             print("after send_binary_plane BLOCK", flush=True)
#         else:
#             print("before send_image_mono BLOCK", flush=True)
#             dmd.send_image_mono(BLOCK_PLANE, block_mask)
#             print("after send_image_mono BLOCK", flush=True)

#         print("before show_plane BLOCK", flush=True)
#         dmd.show_plane(BLOCK_PLANE)
#         print("after show_plane BLOCK", flush=True)

#         print("BLOCK state should be visible now.")
#         time.sleep(2)

#         print("Test completed successfully.", flush=True)

#     finally:
#         if dmd is not None:
#             print("Disconnecting DMD...", flush=True)
#             try:
#                 dmd.disconnect()
#                 print("Disconnected.", flush=True)
#             except Exception as exc:
#                 print("Disconnect failed:", repr(exc), flush=True)


# if __name__ == "__main__":
#     main()



import labrad

cxn = labrad.connect(username="", password="")
dmd = cxn.dmd_dlp6500

print(dmd.get_state())

input("Press Enter to PASS all...")
dmd.pass_all(False)


input("Press Enter to BLOCK all...")
dmd.block_all()

input("Press Enter to PASS all again...")
dmd.pass_all(False)


# print("Initial state:")
# print(dmd.get_state())
# input("Press Enter to initialize pass_zero_block...")

# # Loads final chain file and applies pass_zero_block.
# # After this, zero order should be BLOCKED.
# print("initialize_pass_state:")
# print(dmd.initialize_pass_state())
# input("Zero order should now be BLOCKED. Press Enter to turn zero order ON...")

# # Zero-order ON / visible
# print("Turning zero block OFF: zero order should be visible/pass.")
# dmd.zero_block_off()
# print(dmd.get_state())
# input("Look at camera. Zero order should be ON/visible. Press Enter to block it again...")

# # Zero-order OFF / blocked
# print("Turning zero block ON: zero order should be blocked.")
# dmd.zero_block_on()
# print(dmd.get_state())
# input("Look at camera. Zero order should be OFF/blocked. Press Enter to finish...")

# # Leave system in normal safe state: pass everything except zero order.
# dmd.zero_block_on()
# print("Finished. Left DMD in pass_zero_block state.")