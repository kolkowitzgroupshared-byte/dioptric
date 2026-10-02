# -*- coding: utf-8 -*-
"""
V37 Johnson multifield B-field provenance and reconstruction audit.

This script freezes the historically recovered ODMR provenance for the
49/59/62/65 G Johnson spin-echo datasets.  It intentionally distinguishes:
  * raw resonance acquisitions,
  * historical ODMR quartets,
  * the locked reference orientation permutation,
  * later ESEEM-catalog metadata,
  * current V32 B vectors.

No C13 fitting is performed here.  This is the provenance layer that should
be trusted before empirical B-field uncertainty/profile fitting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.b_field_and_coils_calcaultions.sc_b_field_calculations import (
    solve_B_from_odmr_order_invariant,
    solve_B_with_fixed_perm,
)

D_GHZ = 2.8785
GAMMA_E_MHZ_PER_G = 2.8025

OUT_DIR = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo"
    r"\c13_spin_echo_v37_bfield_profile_multifield\2026_09"
)

F_REF_49 = np.array([2.7666, 2.7851, 2.8222, 2.8406], float)

RECORDS = [
    dict(
        label="49G_original_odmr",
        physical_field="49G",
        resonance_primary="2025_11_01-07_35_08-johnson-nv0_2025_10_21",
        resonance_companion="2025_11_09-10_40_49-johnson-nv0_2025_10_21",
        resonance_path=r"G:\nvdata\pc_Purcell\branch_master\resonance\2025_11",
        quartet=[2.7666, 2.7851, 2.8222, 2.8406],
        solve_mode="order_invariant_reference",
        git_resonance_or_b_commit="0f26bb977",
        git_note="b vector field esimation; sc_resonance_analysis.py active on Nov-01 204-NV resonance",
        expected_current_v32=None,
        note="Original Nov-04 wiki B reconstruction.",
    ),
    dict(
        label="49G_catalog_direction",
        physical_field="49G",
        resonance_primary="2025_11_01-07_35_08-johnson-nv0_2025_10_21",
        resonance_companion="2025_11_09-10_40_49-johnson-nv0_2025_10_21",
        resonance_path=r"G:\nvdata\pc_Purcell\branch_master\resonance\2025_11",
        quartet=[2.7665, 2.7846, 2.8230, 2.8410],
        solve_mode="fixed_reference_perm",
        git_resonance_or_b_commit="1b599358b",
        git_note="Nov-04 C13 simulation commit changed B_vec_G to the later 49-G direction",
        expected_current_v32=[-46.19581364, -17.44900422, -5.57935388],
        note=(
            "Rounded quartet reconstructs the direction used by the 49-G ESEEM catalog. "
            "Catalog itself used gamma_C13=10.708 MHz/T."
        ),
    ),
    dict(
        label="65G_rough",
        physical_field="65G",
        resonance_primary="2025_11_21-06_06_26-johnson-nv0_2025_10_21",
        resonance_companion="2025_11_20-09_14_44-johnson-nv0_2025_10_21",
        resonance_path=r"G:\nvdata\pc_Purcell\branch_master\resonance\2025_11",
        quartet=[2.7245, 2.7471, 2.8480, 2.8282],
        solve_mode="fixed_reference_perm",
        git_resonance_or_b_commit="72cfc7672",
        git_note="spin echo at 65G; introduced solve_B_with_fixed_perm and this f_new quartet",
        expected_current_v32=[-31.61263115, -56.58135644, -6.55120020],
        note="Nov-20 is 312-NV companion/all-orientation scan; Nov-21 is 204-NV resonance.",
    ),
    dict(
        label="65G_refined",
        physical_field="65G",
        resonance_primary="2025_11_21-06_06_26-johnson-nv0_2025_10_21",
        resonance_companion="2025_11_20-09_14_44-johnson-nv0_2025_10_21",
        resonance_path=r"G:\nvdata\pc_Purcell\branch_master\resonance\2025_11",
        quartet=[2.7252, 2.7464, 2.8487, 2.8275],
        solve_mode="fixed_reference_perm",
        git_resonance_or_b_commit="17d7d6b96",
        git_note="new b field updates; later orientation-locked 65-G quartet",
        expected_current_v32=[-31.61263115, -56.58135644, -6.55120020],
        note="Refined quartet is internally exact under the linear projection model.",
    ),
    dict(
        label="59G",
        physical_field="59G",
        resonance_primary="2025_11_29-04_02_02-johnson-nv0_2025_10_21",
        resonance_companion="2025_11_28-01_53_35-johnson-nv0_2025_10_21",
        resonance_path=r"G:\nvdata\pc_Purcell\branch_master\resonance\2025_11",
        quartet=[2.7081, 2.8083, 2.8251, 2.8536],
        solve_mode="fixed_reference_perm",
        git_resonance_or_b_commit="76311b534",
        git_note="eseem analysis updates; final 59-G B introduced while Nov-29 resonance was the active historical scan",
        expected_current_v32=[-41.57848995, -32.77145194, -27.57993480],
        note="Nov-28 is the 312-NV 3A,0 companion; Nov-29 is the 204-NV scan.",
    ),
    dict(
        label="62G",
        physical_field="62G",
        resonance_primary="2025_12_20-06_01_33-johnson-nv0_2025_10_21",
        resonance_companion="2025_12_10-10_28_25-johnson-nv0_2025_10_21",
        resonance_path=r"G:\nvdata\pc_Purcell\branch_master\resonance\2025_12",
        quartet=[2.7859, 2.7098, 2.8706, 2.8169],
        solve_mode="fixed_reference_perm",
        git_resonance_or_b_commit="13f98b18f / 6e5e1a4fb",
        git_note="Dec-20 raw/new-field commit; Dec-24 62-G C13 catalog commit",
        expected_current_v32=[-48.67047318, -32.07615947, 22.49657427],
        note=(
            "Independent order-invariant solve can choose the symmetry-equivalent -Bz branch. "
            "Locked 49-G line->NV permutation gives +Bz and exactly matches the current catalog convention."
        ),
    ),
]

def arrstr(a):
    return json.dumps([float(x) for x in np.asarray(a, float)])

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = solve_B_from_odmr_order_invariant(
        F_REF_49, D_GHz=D_GHZ, gamma_e_MHz_per_G=GAMMA_E_MHZ_PER_G
    )
    perm_ref = tuple(ref["perm"])

    rows = []
    details = {}

    for rec in RECORDS:
        f = np.asarray(rec["quartet"], float)
        if rec["solve_mode"] == "order_invariant_reference":
            sol = solve_B_from_odmr_order_invariant(
                f, D_GHz=D_GHZ, gamma_e_MHz_per_G=GAMMA_E_MHZ_PER_G
            )
        else:
            sol = solve_B_with_fixed_perm(
                f,
                perm_ref,
                D_GHz=D_GHZ,
                gamma_e_MHz_per_G=GAMMA_E_MHZ_PER_G,
            )

        B = np.asarray(sol["B"], float)
        expected = rec["expected_current_v32"]
        delta = np.nan
        if expected is not None:
            delta = float(np.linalg.norm(B - np.asarray(expected, float)))

        row = dict(
            label=rec["label"],
            physical_field=rec["physical_field"],
            resonance_primary=rec["resonance_primary"],
            resonance_companion=rec["resonance_companion"],
            resonance_directory=rec["resonance_path"],
            git_commit=rec["git_resonance_or_b_commit"],
            solve_mode=rec["solve_mode"],
            quartet_GHz=arrstr(f),
            locked_perm=str(perm_ref),
            Bx_G=float(B[0]),
            By_G=float(B[1]),
            Bz_G=float(B[2]),
            Bmag_G=float(sol["B_mag"]),
            residual_norm_G=float(sol["residual_norm"]),
            rms_projection_mismatch_G=float(sol["residual_norm"]) / 2.0,
            signs=str(tuple(int(x) for x in sol["signs"])),
            delta_to_current_v32_G=delta,
            git_note=rec["git_note"],
            note=rec["note"],
        )
        rows.append(row)
        details[rec["label"]] = {
            **rec,
            "solution": {
                "B_G": B.tolist(),
                "Bmag_G": float(sol["B_mag"]),
                "B_hat": np.asarray(sol["B_hat"], float).tolist(),
                "residual_norm_G": float(sol["residual_norm"]),
                "rms_projection_mismatch_G": float(sol["residual_norm"]) / 2.0,
                "perm": list(sol["perm"]),
                "signs": np.asarray(sol["signs"], int).tolist(),
                "f_minus_pred_nvaxes_GHz": np.asarray(
                    sol["f_minus_nvaxes_GHz"], float
                ).tolist(),
            },
        }

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v37_bfield_provenance.csv", index=False)

    # 49-G gamma bookkeeping audit.
    fI_catalog_Hz = 53199.11262026196
    gamma_catalog_kHz_per_G = 1.0708
    gamma_later_kHz_per_G = 1.0705
    Bmag_catalog_native = fI_catalog_Hz / (1000.0 * gamma_catalog_kHz_per_G)
    Bmag_if_inverted_with_10705 = fI_catalog_Hz / (1000.0 * gamma_later_kHz_per_G)

    B49_direction = np.array([-46.18287122, -17.44411563, -5.57779074], float)
    B49_hat = B49_direction / np.linalg.norm(B49_direction)
    gamma_audit = {
        "catalog_fI_Hz": fI_catalog_Hz,
        "catalog_gamma_kHz_per_G": gamma_catalog_kHz_per_G,
        "catalog_native_Bmag_G": Bmag_catalog_native,
        "later_gamma_kHz_per_G": gamma_later_kHz_per_G,
        "Bmag_if_catalog_fI_inverted_with_later_gamma_G": Bmag_if_inverted_with_10705,
        "B_native_from_catalog_gamma_G": (B49_hat * Bmag_catalog_native).tolist(),
        "B_rescaled_if_10705_G": (B49_hat * Bmag_if_inverted_with_10705).tolist(),
        "interpretation": (
            "The current ~49.6955746 G V32 vector is the 49.6816517 G catalog "
            "direction rescaled by using gamma_C13=1.0705 kHz/G to invert a catalog "
            "fI generated with gamma_C13=1.0708 kHz/G. It is not an independent ODMR measurement."
        ),
    }

    payload = {
        "reference_quartet_GHz": F_REF_49.tolist(),
        "reference_locked_perm": list(perm_ref),
        "records": details,
        "gamma_49G_audit": gamma_audit,
    }
    with open(OUT_DIR / "v37_bfield_provenance.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines = [
        "V37 JOHNSON MULTIFIELD B-FIELD PROVENANCE",
        "==========================================",
        "",
        f"Locked reference permutation (49 G): {perm_ref}",
        "",
        "Primary conclusion:",
        "  Use a common reference line->NV-axis permutation across fields.",
        "  Do not independently canonicalize each four-line quartet because symmetry-equivalent",
        "  crystal-frame branches can relabel axes/signs (especially visible for 62 G).",
        "",
        "49 G gamma bookkeeping:",
        f"  catalog fI = {fI_catalog_Hz:.9f} Hz",
        f"  native catalog gamma = {gamma_catalog_kHz_per_G:.4f} kHz/G -> |B|={Bmag_catalog_native:.9f} G",
        f"  later inversion gamma = {gamma_later_kHz_per_G:.4f} kHz/G -> |B|={Bmag_if_inverted_with_10705:.9f} G",
        "  Therefore the 49.6956-G value is a gamma-constant conversion artifact, not a new field measurement.",
        "",
        "Historical reconstructions:",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"  {r['label']}: B=({r.Bx_G:.8f},{r.By_G:.8f},{r.Bz_G:.8f}) G, "
            f"|B|={r.Bmag_G:.8f} G, residual={r.residual_norm_G:.6f} G, "
            f"raw={r.resonance_primary}, commit={r.git_commit}"
        )
    lines += [
        "",
        "Important:",
        "  The residuals here quantify consistency of the manually recorded four-line quartets",
        "  with the simple linear ODMR projection model. They are NOT statistical confidence intervals.",
        "  V37 should next refit the raw resonance scans and propagate empirical line-center uncertainty.",
    ]
    (OUT_DIR / "README_PROVENANCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(df[[
        "label","Bmag_G","Bx_G","By_G","Bz_G","residual_norm_G",
        "delta_to_current_v32_G","resonance_primary","git_commit"
    ]].to_string(index=False))
    print("\n49G gamma audit:", json.dumps(gamma_audit, indent=2))
    print("\nSaved to", OUT_DIR)

if __name__ == "__main__":
    main()
