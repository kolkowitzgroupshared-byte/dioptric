import shutil
import os
from pathlib import Path
import numpy as np


def update_nv_blob_coords_by_first_nv_shift(
    file_path="slmsuite/nv_blob_detection/nv_blob_1274nvs_reordered.npz",
    old_first=(194.989, 194.924),
    new_first=(194.039, 189.963),
    overwrite=True,
):
    file_path = Path(file_path)

    old_first = np.asarray(old_first, dtype=np.float32)
    new_first = np.asarray(new_first, dtype=np.float32)
    delta = new_first - old_first

    print("Old first coord:", old_first)
    print("New first coord:", new_first)
    print("Applying delta to all NV coordinates:", delta)

    # IMPORTANT: use context manager so the npz file is closed before overwrite.
    with np.load(file_path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}

    if "nv_coordinates" not in out:
        raise KeyError(f"{file_path} does not contain key 'nv_coordinates'.")

    nv_coordinates_old = np.asarray(out["nv_coordinates"], dtype=np.float32)
    nv_coordinates_new = nv_coordinates_old + delta[None, :]

    out["nv_coordinates"] = nv_coordinates_new

    print("Number of NVs:", len(nv_coordinates_new))
    print("First coord before:", nv_coordinates_old[0])
    print("First coord after: ", nv_coordinates_new[0])

    backup_path = file_path.with_suffix(".backup_before_shift.npz")
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        print("Backup saved to:", backup_path)
    else:
        print("Backup already exists:", backup_path)

    if overwrite:
        tmp_path = file_path.with_name(file_path.stem + "_tmp_shifted.npz")
        np.savez_compressed(tmp_path, **out)

        # os.replace is the safest overwrite method on Windows,
        # but it still fails if the destination file is open elsewhere.
        os.replace(tmp_path, file_path)
        print("Updated file overwritten:", file_path)
    else:
        out_path = file_path.with_name(file_path.stem + "_shifted.npz")
        np.savez_compressed(out_path, **out)
        print("Updated file saved to:", out_path)

    return nv_coordinates_new, delta


if __name__ == "__main__":
    update_nv_blob_coords_by_first_nv_shift()