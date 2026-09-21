# -*- coding: utf-8 -*-
"""
Build the 52 G orientation-aware ESEEM catalog using Saroj's established
exact-kappa workflow in analysis/spin_echo_work/kappa_modulation_depth.py.

Run from the dioptric repository root:

    python analysis/spin_echo_work/build_essem_catalog_52G_exact_kappa.py

Outputs:
    analysis/spin_echo_work/essem_freq_kappa_catalog_22A_52G.json
    analysis/spin_echo_work/essem_freq_kappa_catalog_22A_52G.csv
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.spin_echo_work.kappa_modulation_depth import (
    build_essem_catalog_with_kappa,
)


# =============================================================================
# FIELD / PHYSICS
# =============================================================================

# ODMR reconstruction: all-negative sign branch requested for this dataset.
B_VECTOR_G = np.array(
    [-48.551229, -18.748242, -5.973533],
    dtype=float,
)
B_VECTOR_T = B_VECTOR_G * 1e-4

GAMMA_C13_HZ_PER_T = 10.705e6
P_C13 = 0.011
MS = -1

ORIENTATIONS = (
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
)

HYPERFINE_PATH = Path(
    r"analysis\nv_hyperfine_coupling\nv-2.txt"
)

DISTANCE_MAX_A = 22.0
PHI_DEG = 0.0

OUT_JSON = Path(
    r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json"
)
OUT_CSV = Path(
    r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.csv"
)


def main():
    if not HYPERFINE_PATH.exists():
        raise FileNotFoundError(
            f"Hyperfine table not found: {HYPERFINE_PATH}"
        )

    OUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    B_mag_G = float(
        np.linalg.norm(B_VECTOR_G)
    )

    larmor_kHz = (
        GAMMA_C13_HZ_PER_T
        * np.linalg.norm(B_VECTOR_T)
        / 1e3
    )

    revival_tau_us = (
        1000.0 / larmor_kHz
    )

    print("=" * 78)
    print("BUILDING 52 G EXACT-KAPPA ESEEM CATALOG")
    print("=" * 78)
    print(f"B vector (G):      {B_VECTOR_G.tolist()}")
    print(f"|B| (G):           {B_mag_G:.6f}")
    print(f"13C Larmor (kHz):  {larmor_kHz:.6f}")
    print(f"Revival tau (us):  {revival_tau_us:.6f}")
    print(f"Revival 2tau (us): {2.0 * revival_tau_us:.6f}")
    print(f"Hyperfine table:   {HYPERFINE_PATH}")
    print()

    records = build_essem_catalog_with_kappa(
        hyperfine_path=str(HYPERFINE_PATH),
        B_lab_vec=B_VECTOR_T,
        orientations=ORIENTATIONS,
        distance_max_A=DISTANCE_MAX_A,
        gamma_n_Hz_per_T=GAMMA_C13_HZ_PER_T,
        p_occ=P_C13,
        ms=MS,
        phi_deg=PHI_DEG,
        out_json=str(OUT_JSON),
        out_csv=str(OUT_CSV),
        read_hf_table_fn=None,
    )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    if not records:
        raise RuntimeError(
            "Catalog builder returned no records."
        )

    required = {
        "orientation",
        "site_index",
        "distance_A",
        "kappa",
        "f_minus_Hz",
        "f_plus_Hz",
        "fI_Hz",
        "line_w_minus",
        "line_w_plus",
    }

    missing = (
        required
        - set(records[0].keys())
    )

    if missing:
        raise RuntimeError(
            f"Catalog is missing required fields: {sorted(missing)}"
        )

    orientation_counts = Counter(
        tuple(r["orientation"])
        for r in records
    )

    kappas = np.asarray(
        [float(r["kappa"]) for r in records],
        dtype=float,
    )

    fI_kHz = np.asarray(
        [float(r["fI_Hz"]) / 1e3 for r in records],
        dtype=float,
    )

    fminus_kHz = np.asarray(
        [float(r["f_minus_Hz"]) / 1e3 for r in records],
        dtype=float,
    )

    fplus_kHz = np.asarray(
        [float(r["f_plus_Hz"]) / 1e3 for r in records],
        dtype=float,
    )

    print()
    print("=" * 78)
    print("CATALOG VALIDATION")
    print("=" * 78)
    print(f"Total records:      {len(records)}")

    for ori in ORIENTATIONS:
        print(
            f"  {ori}: "
            f"{orientation_counts.get(tuple(ori), 0)} sites"
        )

    print(
        f"kappa range:        "
        f"{np.nanmin(kappas):.6g} to "
        f"{np.nanmax(kappas):.6g}"
    )
    print(
        f"median fI:          "
        f"{np.nanmedian(fI_kHz):.6f} kHz"
    )
    print(
        f"f- range:           "
        f"{np.nanmin(fminus_kHz):.3f} to "
        f"{np.nanmax(fminus_kHz):.3f} kHz"
    )
    print(
        f"f+ range:           "
        f"{np.nanmin(fplus_kHz):.3f} to "
        f"{np.nanmax(fplus_kHz):.3f} kHz"
    )
    print(f"JSON:               {OUT_JSON}")
    print(f"CSV:                {OUT_CSV}")
    print("=" * 78)


if __name__ == "__main__":
    main()
