# -*- coding: utf-8 -*-
"""
Extract the magnetic field (crystallographic axes only)
Created on March 23th, 2025
@author: Saroj Chand
"""

import numpy as np
from itertools import product, permutations

# ----------------- NV geometry -----------------
def nv_axes():
    """Four NV <111> unit vectors in the cubic {x,y,z} basis."""
    axes = np.array([
        [ 1,  1,  1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1,  1, -1],
    ], dtype=float)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes

# -------- Solve B from only 3 ODMR lines (predict the 4th) ----------

def solve_B_from_three_odmr(
    f3_GHz,
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    sign_eps=1e-12,
    round_decimals=12
):
    """
    Order-invariant 3-line solver in crystal axes.
    Tries: which NV is missing (4 choices) × permutations of provided 3 (6) × sign patterns (2^3).
    Returns B_after, predicted 4th line, and full mapping.

    f3_GHz: list/array of 3 lower-branch frequencies (GHz), any order.
    """
    f3 = np.asarray(f3_GHz, float).reshape(3)
    n_all = nv_axes()                 # (4,3)
    NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]

    # Convert to |n·B| from the 3 lines
    abs_nB_meas_3 = (D_GHz - f3) * 1000.0 / float(gamma_e_MHz_per_G)

    best = None
    # choose which orientation is missing
    for missing_idx in range(4):
        # the 3 orientations we assign to the provided lines
        ori_idxs = [i for i in range(4) if i != missing_idx]
        n3 = n_all[ori_idxs, :]  # (3,3)

        for perm in permutations(range(3)):  # order of mapping f3 -> ori_idxs
            b_mag = abs_nB_meas_3[list(perm)]  # (3,)
            for s in product([-1.0, +1.0], repeat=3):
                s = np.asarray(s, float)
                rhs = s * b_mag  # target signed projections for the chosen 3 orientations

                # Solve n3 B = rhs (3x3)
                try:
                    B = np.linalg.solve(n3, rhs)
                except np.linalg.LinAlgError:
                    continue

                proj = n3 @ B
                # robust sign-consistency for the used 3
                if not np.all(proj * rhs >= -sign_eps):
                    continue

                # residual on the used 3
                res = float(np.linalg.norm(proj - rhs))

                # predict the missing orientation’s projection and f_-
                proj_missing = float(n_all[missing_idx] @ B)
                abs_proj_missing = abs(proj_missing)
                f_missing = D_GHz - (gamma_e_MHz_per_G * abs_proj_missing) / 1000.0

                # canonical tie-break: lexicographically smallest rounded B
                key = (np.round(B, round_decimals).tolist(), res, missing_idx, perm, s.tolist())
                cand = dict(
                    B=B, res=res, missing_idx=missing_idx, perm=perm, signs=s,
                    proj3=proj, ori_idxs=ori_idxs, f_missing=f_missing,
                    abs_missing_G=abs_proj_missing, proj_missing=proj_missing
                )
                if (best is None) or (key < best["_key"]):
                    cand["_key"] = key
                    best = cand

    if best is None:
        raise RuntimeError("No consistent 3-line solution found; check inputs.")

    # Build readable outputs
    B = best["B"]
    B_mag = float(np.linalg.norm(B))
    B_hat = B / B_mag if B_mag > 0 else np.zeros_like(B)

    # Predicted full quartet (nv_axes order)
    projs_all = n_all @ B
    abs_all = np.abs(projs_all)
    f_all = D_GHz - (gamma_e_MHz_per_G * abs_all) / 1000.0

    # Map the provided 3 lines back to NV labels
    mapping = []
    for j_meas in range(3):
        i_ori = best["ori_idxs"][best["perm"][j_meas]]
        mapping.append(dict(
            idx_meas=j_meas,
            f_GHz=f3[j_meas],
            ori_idx=i_ori,
            ori_label=NV_LABELS[i_ori],
            n_dot_B=projs_all[i_ori],
            abs_nB_meas_G=abs_nB_meas_3[j_meas],
            abs_nB_pred_G=abs_all[i_ori],
            sign=int(np.sign(projs_all[i_ori])) if abs_all[i_ori] > 0 else 0
        ))

    return {
        "B": B,
        "B_mag": B_mag,
        "B_hat": B_hat,
        "f_minus_nvaxes_GHz": f_all,           # predicted 4-tuple in nv_axes order
        "projections": projs_all,              # n·B for all 4 (nv_axes order)
        "abs_proj_G": abs_all,                 # |n·B| (G) for all 4
        "missing_ori_idx": best["missing_idx"],
        "missing_ori_label": NV_LABELS[best["missing_idx"]],
        "predicted_missing_f_GHz": best["f_missing"],
        "mapping_meas3": mapping,
        "residual_norm": best["res"],
    }

def print_three_line_summary(f3, D_GHz, gamma_e_MHz_per_G, out3):
    NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]
    print("\n========== 3-line NV ODMR → B solve (crystal frame) ==========")
    print("Input 3× f_- (GHz), any order:", np.round(f3, 6))
    print("\n--- Magnetic field (crystal axes) ---")
    print("B (G):", np.round(out3["B"], 9))
    print("|B| (G):", f"{out3['B_mag']:.12f}")
    print("B_hat:", np.round(out3["B_hat"], 9))

    print("\n--- Predicted full quartet in nv_axes order (GHz) ---")
    for i in range(4):
        print(f" NV {NV_LABELS[i]}: f_- = {out3['f_minus_nvaxes_GHz'][i]:.9f} GHz")

    print("\n--- Which NV is missing? ---")
    print(f" Missing orientation: NV {out3['missing_ori_label']} (idx {out3['missing_ori_idx']})")
    print(f" Predicted missing f_-: {out3['predicted_missing_f_GHz']:.9f} GHz")

    print("\n--- Mapping for the 3 provided lines (measured order) ---")
    for m in out3["mapping_meas3"]:
        print(f" meas idx {m['idx_meas']}: f_-={m['f_GHz']:.9f} GHz  →  NV {m['ori_label']} "
              f"(ori idx {m['ori_idx']});  n·B={m['n_dot_B']:.6f} G, "
              f"|n·B|_meas={m['abs_nB_meas_G']:.6f} G, |n·B|_pred={m['abs_nB_pred_G']:.6f} G, "
              f"sign={m['sign']:>+d}")

    print("\n--- Fit quality ---")
    print(f"Residual norm (3 used eqs): {out3['residual_norm']:.3e}")
    print("============================================================\n")

# ----------------- Example -----------------
if __name__ == "__main__":
    # try any ordering you like:    
    D = 2.8785
    gamma = 2.8025
    # Your three after-lines (any order)
    # f_after3 = [2.7991, 2.8282, 2.8697]
    # f_after3  = [2.7829, 2.771367, 2.8202, 2.810689] ### Coils off
    f_after3  = [2.7829, 2.771367,  2.810689] ### Coils off
    # f_after3  = [2.7829, 2.771367, 2.8202] ### Coils off

    out3 = solve_B_from_three_odmr(f_after3, D_GHz=D, gamma_e_MHz_per_G=gamma)
    print_three_line_summary(f_after3, D_GHz=D, gamma_e_MHz_per_G=gamma, out3=out3)

    # Optional: compare to your baseline B0 from the 4-line solve to get ΔB
    # (Assuming you computed out0 from the full baseline quartet earlier)
    # dB = out3["B"] - out0["B"]
    # print("ΔB (G) from this setting vs baseline:", np.round(dB, 6))
    # print("Δf (MHz) in nv_axes order:",
    #       np.round(1000*(out0["f_minus_nvaxes_GHz"] - out3["f_minus_nvaxes_GHz"]), 6))

