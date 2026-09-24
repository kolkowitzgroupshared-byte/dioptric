# -*- coding: utf-8 -*-
"""Promote a validated Purcell calibration into the offline-safe current set."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PURCELL_ROOT = Path(__file__).resolve().parent
CURRENT_ROOT = PURCELL_ROOT / "current"
MANIFEST_PATH = PURCELL_ROOT / "manifest.json"
HISTORY_ROOT = Path(r"G:\nvdata\pc_Purcell\calibration_history")

ROLE_TO_FILE = {
    "active_nv_coords": "active_nv_coords.npz",
    "slm_fourier": "slm_fourier.h5",
    "nuvu_to_thorcam_slm": "nuvu_to_thorcam_slm.npz",
    "dmd_zero_order": "dmd_zero_order.npz",
    "dmd_triangle_affine": "dmd_triangle_affine.npz",
    "nuvu_to_thorcam_dmd": "nuvu_to_thorcam_dmd.npz",
    "dmd_nv_chain": "dmd_nv_chain.npz",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def promote(role, source):
    source = Path(source).resolve()
    if role not in ROLE_TO_FILE:
        raise ValueError(f"Unknown role: {role}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not HISTORY_ROOT.parent.exists():
        raise RuntimeError(
            "G:\\nvdata is not available. Promotion is intentionally blocked "
            "so the previous current calibration can be backed up first."
        )

    CURRENT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = CURRENT_ROOT / ROLE_TO_FILE[role]
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    backup_dir = HISTORY_ROOT / datetime.now().strftime("%Y_%m") / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    previous_hash = None
    backup_path = None
    if destination.exists():
        previous_hash = sha256(destination)
        backup_path = backup_dir / f"previous_{destination.name}"
        shutil.copy2(destination, backup_path)
        if sha256(backup_path) != previous_hash:
            raise RuntimeError("Backup hash verification failed.")

    candidate_hash = sha256(source)
    history_path = backup_dir / f"promoted_{destination.name}"
    shutil.copy2(source, history_path)
    if sha256(history_path) != candidate_hash:
        raise RuntimeError("History copy hash verification failed.")

    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temp_path)
    if sha256(temp_path) != candidate_hash:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Local candidate hash verification failed.")

    os.replace(temp_path, destination)
    current_hash = sha256(destination)
    if current_hash != candidate_hash:
        raise RuntimeError("Promoted current file hash verification failed.")

    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    manifest.setdefault("schema_version", 1)
    manifest["setup"] = "purcell"
    manifest["current_root"] = "calibrations/purcell/current"
    manifest["history_root"] = str(HISTORY_ROOT)
    manifest["last_promotion"] = {
        "timestamp": timestamp,
        "role": role,
        "source": str(source),
        "source_sha256": candidate_hash,
        "current_file": destination.name,
        "current_sha256": current_hash,
        "previous_sha256": previous_hash,
        "previous_backup": str(backup_path) if backup_path else None,
        "history_copy": str(history_path),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Promoted {role}")
    print(f"Current: {destination}")
    print(f"SHA256:  {current_hash}")
    print(f"History: {history_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=sorted(ROLE_TO_FILE))
    parser.add_argument("source", help="Validated calibration file to promote")
    args = parser.parse_args()
    promote(args.role, args.source)


if __name__ == "__main__":
    main()
