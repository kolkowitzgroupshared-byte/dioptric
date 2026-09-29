"""Canonical filesystem roots for spin-echo analysis outputs.

All derived spin-echo results belong under:
    G:\\nvdata\\pc_<host>\\branch_<branch>\\spin_echo\\<analysis_name>\\YYYY_MM

The helper also exposes fixed NVOffice paths used by historical analysis scripts.
Raw-data retrieval through data_manager.get_raw_data() is unchanged.
"""
from pathlib import Path

NV_OFFICE_BRANCH_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master")
SPIN_ECHO_ROOT = NV_OFFICE_BRANCH_ROOT / "spin_echo"

NV_OFFICE_UNC_BRANCH_ROOT = Path(
    r"\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master"
)
SPIN_ECHO_UNC_ROOT = NV_OFFICE_UNC_BRANCH_ROOT / "spin_echo"


def result_root(name: str, month: str | None = None) -> Path:
    """Return the canonical root for one spin-echo analysis family."""
    path = SPIN_ECHO_ROOT / str(name)
    if month is not None:
        path = path / str(month)
    return path


def legacy_result_root(name: str, month: str | None = None) -> Path:
    """Legacy pre-migration location, retained only for transition checks."""
    path = NV_OFFICE_BRANCH_ROOT / str(name)
    if month is not None:
        path = path / str(month)
    return path


def existing_or_canonical(name: str, month: str | None = None) -> Path:
    """Prefer canonical location; fall back to a still-unmigrated legacy folder."""
    new = result_root(name, month)
    if new.exists():
        return new
    old = legacy_result_root(name, month)
    return old if old.exists() else new
