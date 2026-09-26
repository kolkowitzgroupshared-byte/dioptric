# -*- coding: utf-8 -*-
"""Build the 52 G per-NV orientation map from QNami ODMR resonance fits.

Authoritative orientation candidates are restricted to the two resonance groups
used for the 212-NV spin-echo subset:
    (1, 1, -1)  <-> 2.7773 / 2.9758 GHz
    (-1, 1, 1)  <-> 2.8421 / 2.9195 GHz

Rule:
  1) use Sep-22 fit if its pair RMS distance to an allowed target <= cutoff;
  2) otherwise Sep-17;
  3) otherwise Sep-14 parent-631 fit;
  4) otherwise choose the closest of those three and flag confidence='low'.

The Sep-17 selected list preserves original 631-NV names.  The Sep-22 list is
the renumbered 212-NV working list.  Their pixel coordinates are required to
match exactly before writing the map.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESONANCE_DIR = Path(r"G:\nvdata\pc_Purcell\branch_master\resonance\2026_09")
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "analysis" / ".resonance_fit_cache"

SEP14_STEM = "2026_09_14-00_27_42-qnami-nv0_2026_02_20"
SEP17_STEM = "2026_09_17-11_59_59-qnami-nv0_2026_02_20"
SEP22_STEM = "2026_09_22-09_09_25-qnami-nv0_2026_02_20"

SEP14_CACHE = CACHE_DIR / "resonance_fit_507b9a65510126a6.npz"
SEP17_CACHE = CACHE_DIR / "resonance_fit_f4d91e38b533e1fa.npz"
SEP22_CACHE = CACHE_DIR / "resonance_fit_3e5541cd49861d39.npz"

TARGETS = {
    "group_A": {
        "orientation": (1, 1, -1),
        "centers_GHz": np.array([2.7773, 2.9758], float),
    },
    "group_B": {
        "orientation": (-1, 1, 1),
        "centers_GHz": np.array([2.8421, 2.9195], float),
    },
}
DEFAULT_CUTOFF_MHZ = 8.0

OUTPUT_CSV = RESONANCE_DIR / "2026_09_22-52G_orientation_map_212_from_resonance.csv"
OUTPUT_JSON = RESONANCE_DIR / "2026_09_22-52G_orientation_map_212_from_resonance.json"


def _load_meta(stem):
    with open(RESONANCE_DIR / f"{stem}.txt", "r", encoding="utf-8") as f:
        return json.load(f)


def _parent_id(name):
    m = re.search(r"nv(\d+)_", str(name))
    if not m:
        raise ValueError(f"Could not parse parent NV id from {name!r}")
    return int(m.group(1))


def _classify(cache, idx):
    z = np.load(cache, allow_pickle=False)
    centers = np.sort(np.asarray(z["params"][idx, 2:4], float))
    distances = {
        label: float(
            np.sqrt(np.mean(((centers - info["centers_GHz"]) * 1000.0) ** 2))
        )
        for label, info in TARGETS.items()
    }
    label = min(distances, key=distances.get)
    other = "group_B" if label == "group_A" else "group_A"
    return {
        "label": label,
        "orientation": TARGETS[label]["orientation"],
        "distance_mhz": distances[label],
        "margin_mhz": distances[other] - distances[label],
        "center_lo_GHz": float(centers[0]),
        "center_hi_GHz": float(centers[1]),
        "red_chi2": float(z["red_chi2"][idx]),
        "success": bool(z["success"][idx]),
    }


def build_map(cutoff_mhz=DEFAULT_CUTOFF_MHZ):
    m17 = _load_meta(SEP17_STEM)
    m22 = _load_meta(SEP22_STEM)
    nv17 = m17["nv_list"]
    nv22 = m22["nv_list"]
    if len(nv17) != 212 or len(nv22) != 212:
        raise RuntimeError(f"Expected 212 NVs, got Sep17={len(nv17)}, Sep22={len(nv22)}")

    xy17 = np.asarray([n["coords"]["pixel"] for n in nv17], float)
    xy22 = np.asarray([n["coords"]["pixel"] for n in nv22], float)
    mapping = []
    for p in xy22:
        d = np.linalg.norm(xy17 - p, axis=1)
        mapping.append(int(np.argmin(d)))
    if len(set(mapping)) != 212:
        raise RuntimeError("Sep22 -> Sep17 coordinate mapping is not one-to-one")
    max_dist = max(float(np.linalg.norm(xy22[i] - xy17[j])) for i, j in enumerate(mapping))
    if max_dist > 1e-6:
        raise RuntimeError(f"Sep22/Sep17 coordinate mismatch: max distance {max_dist}")

    rows = []
    for current_index, j17 in enumerate(mapping):
        parent_nv = nv17[j17]
        current_nv = nv22[current_index]
        parent_index = _parent_id(parent_nv["name"])

        evidence = {
            "sep14": _classify(SEP14_CACHE, parent_index),
            "sep17": _classify(SEP17_CACHE, j17),
            "sep22": _classify(SEP22_CACHE, current_index),
        }

        chosen_source = None
        chosen = None
        for src in ("sep22", "sep17", "sep14"):
            e = evidence[src]
            if e["success"] and e["distance_mhz"] <= float(cutoff_mhz):
                chosen_source = src
                chosen = e
                confidence = "clean"
                break
        if chosen is None:
            good_sources = [s for s in ("sep22", "sep17", "sep14") if evidence[s]["success"]]
            if not good_sources:
                raise RuntimeError(f"NV {current_index}: no successful resonance fits")
            chosen_source = min(good_sources, key=lambda s: evidence[s]["distance_mhz"])
            chosen = evidence[chosen_source]
            confidence = "low"

        row = {
            "nv_index": current_index,
            "current_name": current_nv["name"],
            "parent_631_index": parent_index,
            "parent_631_name": parent_nv["name"],
            "pixel_x": float(xy22[current_index, 0]),
            "pixel_y": float(xy22[current_index, 1]),
            "drift_nv": bool(current_index == 0),
            "group": chosen["label"],
            "orientation": str(tuple(chosen["orientation"])),
            "orientation_source": chosen_source,
            "orientation_confidence": confidence,
            "resonance_cutoff_mhz": float(cutoff_mhz),
        }
        for src, e in evidence.items():
            row[f"{src}_group"] = e["label"]
            row[f"{src}_center_lo_GHz"] = e["center_lo_GHz"]
            row[f"{src}_center_hi_GHz"] = e["center_hi_GHz"]
            row[f"{src}_distance_mhz"] = e["distance_mhz"]
            row[f"{src}_margin_mhz"] = e["margin_mhz"]
            row[f"{src}_red_chi2"] = e["red_chi2"]
            row[f"{src}_success"] = e["success"]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = build_map()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    payload = {
        str(int(r.nv_index)): list(map(int, eval(r.orientation)))
        for r in df.itertuples()
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Wrote:", OUTPUT_CSV)
    print("Wrote:", OUTPUT_JSON)
    print("orientation counts:", df["orientation"].value_counts().to_dict())
    print("confidence:", df["orientation_confidence"].value_counts().to_dict())
    print("source:", df["orientation_source"].value_counts().to_dict())
    print("NV0:", df.iloc[0][[
        "nv_index", "parent_631_index", "drift_nv", "orientation",
        "orientation_source", "orientation_confidence"
    ]].to_dict())


if __name__ == "__main__":
    main()
