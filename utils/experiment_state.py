# -*- coding: utf-8 -*-
"""Authoritative experiment-state loader for Purcell widefield experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import common
from utils import data_manager as dm
from utils import positioning as pos
from utils import widefield
from utils.constants import CoordsKey, NVSig, VirtualLaserKey


DEFAULT_STATE_PATH = Path("calibrations/purcell/experiment_state.json")


def _repo_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else Path(common.get_repo_path()) / path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _indexed_values(mapping, num_nvs, label):
    """Return values ordered by global NV index, requiring exact 0..N-1 coverage."""
    normalized = {int(key): value for key, value in mapping.items()}
    expected = set(range(num_nvs))
    actual = set(normalized)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{label} index mismatch: missing={missing[:10]} extra={extra[:10]} "
            f"(expected {num_nvs} NVs, got {len(actual)} indexed entries)"
        )
    values = [normalized[ind] for ind in range(num_nvs)]
    if any(value is None for value in values):
        bad = [ind for ind, value in enumerate(values) if value is None]
        raise ValueError(f"{label} contains None at NV indices {bad[:20]}")
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise ValueError(f"{label} contains non-finite values at {bad[:20]}")
    return arr


def _validate_calibration_manifest(manifest_path):
    path = _repo_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    failures = []
    for record in manifest.get("files", []):
        current = _repo_path(Path(manifest["current_root"]) / record["current_file"])
        if not current.is_file():
            failures.append(f"missing: {current}")
            continue
        expected_hash = record.get("current_sha256") or record.get("source_sha256")
        if expected_hash and _sha256(current) != expected_hash:
            failures.append(f"hash mismatch: {current}")
    if failures:
        raise RuntimeError("Calibration validation failed: " + "; ".join(failures))
    return manifest


@dataclass
class PurcellExperimentState:
    manifest: dict
    nv_list: list[NVSig]
    pixel_coords: np.ndarray
    green_coords: np.ndarray
    red_coords: np.ndarray
    charge_pol_amps: np.ndarray
    scc_amps: np.ndarray
    scc_durations: np.ndarray
    thresholds: list
    microwaves: dict
    provenance: dict

    @property
    def num_nvs(self):
        return len(self.nv_list)

    @property
    def representative_nv(self):
        return widefield.get_repr_nv_sig(self.nv_list)

    def summary(self):
        return {
            "setup": self.manifest["setup"],
            "sample": self.manifest["sample"]["name"],
            "sample_date": self.manifest["sample"]["sample_date"],
            "num_nvs": self.num_nvs,
            "representative_nv": self.representative_nv.name,
            "threshold_mode": self.manifest["charge_state"]["threshold_mode"],
            "microwave_groups": sorted(self.microwaves),
            "scc_amplitude_source": self.provenance["scc_amplitude_source"],
            "scc_duration_source": self.provenance["scc_duration_source"],
            "nv_coords_sha256": self.provenance["nv_coords_sha256"],
            "calibration_manifest_sha256": self.provenance[
                "calibration_manifest_sha256"
            ],
        }


def load_purcell_state(state_path=DEFAULT_STATE_PATH, sample_overrides=None):
    """Load, resolve, and strictly validate the authoritative Purcell state."""
    state_path = _repo_path(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    if sample_overrides:
        state["sample"] = dict(state["sample"])
        state["sample"].update(sample_overrides)
    config = common.get_config_dict("purcell")

    calibration_manifest_path = state["calibrations"]["manifest_path"]
    _validate_calibration_manifest(calibration_manifest_path)

    nv_path = _repo_path(state["nv_set"]["active_coords_path"])
    with np.load(nv_path, allow_pickle=True) as data:
        pixel_coords = np.asarray(data["nv_coordinates"], dtype=float)
        if "updated_spot_weights" in data:
            spot_weights = np.asarray(data["updated_spot_weights"], dtype=float)
        else:
            spot_weights = None
    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        raise ValueError(f"Invalid NV coordinate shape: {pixel_coords.shape}")
    if not np.all(np.isfinite(pixel_coords)):
        raise ValueError("Active NV coordinates contain non-finite values.")
    num_nvs = len(pixel_coords)
    if spot_weights is not None and len(spot_weights) != num_nvs:
        raise ValueError(
            f"Spot-weight length {len(spot_weights)} != NV count {num_nvs}"
        )

    physical_lasers = config["Optics"]["PhysicalLasers"]
    green_aod = physical_lasers["laser_INTE_520"]["positioner"]
    red_aod = physical_lasers["laser_COBO_638"]["positioner"]

    green_coords = np.asarray(
        [
            pos.transform_coords(coord, CoordsKey.PIXEL, green_aod)
            for coord in pixel_coords
        ],
        dtype=float,
    )
    red_coords = np.asarray(
        [
            pos.transform_coords(coord, CoordsKey.PIXEL, red_aod)
            for coord in pixel_coords
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(green_coords)) or not np.all(np.isfinite(red_coords)):
        raise ValueError("AOD coordinate transform produced non-finite values.")

    charge_pol_amps = np.asarray(
        [widefield.green_qua_amp_fn_2d(coords) for coords in green_coords],
        dtype=float,
    )
    if not np.all(np.isfinite(charge_pol_amps)):
        raise ValueError("Charge-polarization amplitudes contain non-finite values.")

    charge_state = state["charge_state"]
    amp_data = dm.get_raw_data(
        file_stem=charge_state["scc_amplitude_source"], load_npz=True
    )
    scc_amps = _indexed_values(
        amp_data[charge_state["scc_amplitude_key"]], num_nvs, "SCC amplitudes"
    )

    duration_data = dm.get_raw_data(
        file_stem=charge_state["scc_duration_source"], load_npz=True
    )
    scc_durations = _indexed_values(
        duration_data[charge_state["scc_duration_key"]],
        num_nvs,
        "SCC durations",
    )

    threshold_mode = charge_state["threshold_mode"].lower()
    if threshold_mode == "none":
        thresholds = [None] * num_nvs
    else:
        raise NotImplementedError(
            f"Unsupported threshold_mode={threshold_mode!r}; "
            "add an explicit calibrated threshold source before enabling it."
        )

    configured_microwaves = config["Microwaves"]["VirtualSigGens"]
    microwaves = {}
    for group in state.get("microwave_groups", []):
        if group not in configured_microwaves:
            raise ValueError(f"Microwave group {group} is not configured on Purcell.")
        params = dict(configured_microwaves[group])
        required = ("uwave_power", "frequency", "rabi_period", "pi_pulse", "pi_on_2_pulse")
        for key in required:
            if key not in params or not np.isfinite(float(params[key])):
                raise ValueError(f"Microwave group {group} has invalid {key}.")
        microwaves[int(group)] = params

    sample = state["sample"]
    representative_ind = int(state["nv_set"]["representative_nv_index"])
    if not 0 <= representative_ind < num_nvs:
        raise ValueError("representative_nv_index is outside the active NV set.")

    pol_duration = int(charge_state["charge_pol_duration_ns"])
    ion_duration = int(charge_state["ion_duration_ns"])
    nv_list = []
    for ind in range(num_nvs):
        coords = {
            CoordsKey.SAMPLE: list(sample["sample_coords"]),
            CoordsKey.Z: float(sample["z_coord"]),
            CoordsKey.PIXEL: pixel_coords[ind].tolist(),
            green_aod: np.round(green_coords[ind], 3).tolist(),
            red_aod: np.round(red_coords[ind], 3).tolist(),
        }
        nv_list.append(
            NVSig(
                name=f"{sample['name']}-nv{ind}_{sample['sample_date']}",
                coords=coords,
                threshold=thresholds[ind],
                magnet_angle=sample.get("magnet_angle_deg"),
                pulse_durations={
                    VirtualLaserKey.SCC: int(round(scc_durations[ind])),
                    VirtualLaserKey.ION: ion_duration,
                    VirtualLaserKey.CHARGE_POL: pol_duration,
                },
                pulse_amps={
                    VirtualLaserKey.SCC: round(float(scc_amps[ind]), 4),
                    VirtualLaserKey.ION: round(float(scc_amps[ind]), 4),
                    VirtualLaserKey.CHARGE_POL: round(
                        float(charge_pol_amps[ind]), 4
                    ),
                },
            )
        )
    nv_list[representative_ind].representative = True

    calibration_manifest = _repo_path(calibration_manifest_path)
    provenance = {
        "state_path": str(state_path),
        "state_sha256": _sha256(state_path),
        "nv_coords_path": str(nv_path),
        "nv_coords_sha256": _sha256(nv_path),
        "calibration_manifest_path": str(calibration_manifest),
        "calibration_manifest_sha256": _sha256(calibration_manifest),
        "scc_amplitude_source": charge_state["scc_amplitude_source"],
        "scc_duration_source": charge_state["scc_duration_source"],
        "microwave_groups": state.get("microwave_groups", []),
    }

    return PurcellExperimentState(
        manifest=state,
        nv_list=nv_list,
        pixel_coords=pixel_coords,
        green_coords=green_coords,
        red_coords=red_coords,
        charge_pol_amps=charge_pol_amps,
        scc_amps=scc_amps,
        scc_durations=scc_durations,
        thresholds=thresholds,
        microwaves=microwaves,
        provenance=provenance,
    )


def build_manual_reference_nvs(state, pixel_coords, green_coords, red_coords):
    """Build small manual reference NVSigs without altering the active NV array."""
    pixel_coords = np.asarray(pixel_coords, dtype=float)
    green_coords = np.asarray(green_coords, dtype=float)
    red_coords = np.asarray(red_coords, dtype=float)
    if pixel_coords.shape != green_coords.shape or pixel_coords.shape != red_coords.shape:
        raise ValueError("Manual pixel/green/red coordinate arrays must have matching shapes.")
    if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
        raise ValueError("Manual coordinate arrays must have shape (N, 2).")

    sample = state.manifest["sample"]
    manual = []
    for ind in range(len(pixel_coords)):
        manual.append(
            NVSig(
                name=f"{sample['name']}-manual{ind}_{sample['sample_date']}",
                coords={
                    CoordsKey.SAMPLE: list(sample["sample_coords"]),
                    CoordsKey.Z: float(sample["z_coord"]),
                    CoordsKey.PIXEL: pixel_coords[ind].tolist(),
                    "laser_INTE_520_aod": green_coords[ind].tolist(),
                    "laser_COBO_638_aod": red_coords[ind].tolist(),
                },
                magnet_angle=sample.get("magnet_angle_deg"),
            )
        )
    if manual:
        manual[0].representative = True
    return manual


def print_state_summary(state):
    summary = state.summary()
    print("\n=== PURCELL EXPERIMENT STATE ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("validation: OK")
    print("===============================\n")


if __name__ == "__main__":
    resolved = load_purcell_state()
    print_state_summary(resolved)
