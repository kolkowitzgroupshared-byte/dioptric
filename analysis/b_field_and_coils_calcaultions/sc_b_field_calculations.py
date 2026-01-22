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

# Must match the order in nv_axes()
NV_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]


# ----------------- Order-invariant solver -----------------
def solve_B_from_odmr_order_invariant(
    f_ms_minus_GHz,
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    equal_tol=1e-9,
    round_decimals=12,
    sign_eps=1e-12,
):
    """
    Order-invariant NV B-field solver in crystal coordinates.

    Given four m_s = 0 -> -1 ODMR lines f_- (in any order), this:
      - Converts them to |n·B| (G)
      - Brute-forces all 4! permutations (which line -> which NV axis)
      - Brute-forces all 2^4 sign patterns for n·B
      - Solves for B via least squares for each combination
      - Keeps the solutions that are sign-consistent and near-minimal residual
      - Canonicalizes ties by lexicographically smallest rounded B

    Returns a dict with:
      B, B_mag, B_hat,
      projections (n·B in nv_axes order, signed),
      signs (±1 per NV),
      b_magnitudes (|n·B|_meas in nv_axes order),
      abs_nB_pred (|n·B|_pred),
      f_minus_nvaxes_GHz (predicted f_- in nv_axes order),
      residual_norm,
      sign_consistent (always True here),
      perm: tuple (input idx -> nv_axes idx)
    """
    f = np.asarray(f_ms_minus_GHz, float).reshape(4)

    # Guard: m_s=-1 line cannot exceed D (allow tiny FP slack)
    if np.any(f > D_GHz + 1e-9):
        f = np.minimum(f, D_GHz)

    n = nv_axes()  # 4x3

    # |n·B| (G) from measured f_-: |n·B| = (D - f) * 1000 / gamma
    abs_nB_meas = (D_GHz - f) * 1000.0 / float(gamma_e_MHz_per_G)

    best_res = np.inf
    candidates = []
    for perm in permutations(range(4)):
        # impose candidate mapping to nv_axes order
        b_mag = abs_nB_meas[list(perm)]
        for s in product([-1, +1], repeat=4):
            s = np.asarray(s, float)
            rhs = s * b_mag  # signed projections target
            B, *_ = np.linalg.lstsq(n, rhs, rcond=None)
            proj = n @ B
            # robust sign-consistency: proj and rhs must not contradict
            if not np.all(proj * rhs >= -sign_eps):
                continue
            rvec = proj - rhs
            res = float(np.linalg.norm(rvec))
            candidates.append((res, B, proj, s, b_mag, perm))
            if res < best_res:
                best_res = res

    if not candidates:
        raise RuntimeError("No consistent solution found; check inputs.")

    # Keep all solutions with residual within equal_tol of the best,
    # then pick a canonical one by lexicographically smallest rounded B.
    near = [c for c in candidates if c[0] <= best_res + equal_tol]

    def canon_key(c):
        _, B, *_ = c
        return tuple(np.round(B, round_decimals).tolist())

    res, B, proj, s, b_mag, perm = sorted(near, key=canon_key)[0]

    B_mag = float(np.linalg.norm(B))
    B_hat = B / B_mag if B_mag > 0 else np.zeros_like(B)

    # Also provide f_- and |n·B| (predicted) in nv_axes() order
    abs_nB_pred = np.abs(n @ B)
    f_nvaxes = D_GHz - (gamma_e_MHz_per_G * abs_nB_pred) / 1000.0

    return {
        "B": B,
        "B_mag": B_mag,
        "B_hat": B_hat,
        "projections": proj,            # n·B (signed), nv_axes order
        "signs": s,
        "b_magnitudes": b_mag,          # |n·B| from data, nv_axes order (per chosen perm)
        "abs_nB_pred": abs_nB_pred,     # |n·B| from B, nv_axes order (prediction)
        "f_minus_nvaxes_GHz": f_nvaxes, # predicted f_- in nv_axes order
        "residual_norm": res,
        "sign_consistent": True,
        "perm": perm,                   # input idx -> nv_axes idx
    }


# ----------------- Solver with fixed permutation -----------------
def solve_B_with_fixed_perm(
    f_ms_minus_GHz,
    perm_ref,
    D_GHz=2.8785,
    gamma_e_MHz_per_G=2.8025,
    sign_eps=1e-12,
):
    """
    Solve for B assuming a *fixed* mapping from measured lines to NV axes.

    Args:
      f_ms_minus_GHz: array-like length 4, m_s=0 -> -1 ODMR frequencies in
                      arbitrary input order (same order as perm_ref).
      perm_ref: tuple/list of length 4, with perm_ref[in_idx] = nv_axis_idx.
                This is typically obtained once from solve_B_from_odmr_order_invariant
                on a reference dataset, then reused for all subsequent datasets.

    Returns a dict with the same keys as solve_B_from_odmr_order_invariant,
    but with "perm" always equal to perm_ref.
    """
    f = np.asarray(f_ms_minus_GHz, float).reshape(4)
    n = nv_axes()

    # Convert to |n·B| (G) from data (input order)
    abs_nB_meas_in = (D_GHz - f) * 1000.0 / float(gamma_e_MHz_per_G)

    # Reorder into nv_axes order using *fixed* perm_ref
    # perm_ref: input idx -> nv_idx
    b_mag_nv = np.empty(4, float)
    for in_idx, nv_idx in enumerate(perm_ref):
        b_mag_nv[nv_idx] = abs_nB_meas_in[in_idx]

    best_res = np.inf
    best = None
    for s in product([-1, +1], repeat=4):
        s = np.asarray(s, float)
        rhs = s * b_mag_nv
        B, *_ = np.linalg.lstsq(n, rhs, rcond=None)
        proj = n @ B
        if np.any(proj * rhs < -sign_eps):
            continue
        res = float(np.linalg.norm(proj - rhs))
        if res < best_res:
            best_res = res
            best = (B, proj, s, b_mag_nv)

    if best is None:
        raise RuntimeError("No sign-consistent solution for fixed perm_ref.")

    B, proj, s_best, b_mag_best = best
    B_mag = float(np.linalg.norm(B))
    B_hat = B / B_mag if B_mag > 0 else np.zeros_like(B)
    abs_nB_pred = np.abs(proj)
    f_nvaxes = D_GHz - (gamma_e_MHz_per_G * abs_nB_pred) / 1000.0

    return {
        "B": B,
        "B_mag": B_mag,
        "B_hat": B_hat,
        "projections": proj,
        "signs": s_best,
        "b_magnitudes": b_mag_best,
        "abs_nB_pred": abs_nB_pred,
        "f_minus_nvaxes_GHz": f_nvaxes,
        "residual_norm": best_res,
        "sign_consistent": True,
        "perm": tuple(perm_ref),    # always the same now
    }


# ----------------- Pretty printer -----------------
def print_full_summary(f_any_order, out, D_GHz=2.8785, gamma_e_MHz_per_G=2.8025):
    """
    Pretty-print the solution, assuming `out["perm"]` is the authoritative mapping.

    This function does NOT attempt to re-optimize the permutation: it just uses
    out["perm"] to define which input line corresponds to which NV orientation.
    """
    f_any_order = np.asarray(f_any_order, float).reshape(4)
    n = nv_axes()

    print("\n========== NV ODMR → B solve (crystal frame) ==========")
    print("Input f_- (GHz), given order:", f_any_order)
    print("Chosen permutation (input idx → nv_axes idx):", out["perm"])
    print("\n--- Magnetic field (crystal axes) ---")
    print(f"B (G): {out['B']}")
    print(f"|B| (G): {out['B_mag']:.12f}")
    print(f"B_hat: {out['B_hat']}")

    print("\n--- Projections onto NV axes (nv_axes order) ---")
    for i in range(4):
        print(
            f" NV {NV_LABELS[i]}:  n·B = {out['projections'][i]: .6f} G   "
            f"|n·B|_meas = {out['b_magnitudes'][i]: .6f} G   "
            f"|n·B|_pred = {out['abs_nB_pred'][i]: .6f} G   "
            f"sign = {int(out['signs'][i]):+d}"
        )

    print("\n--- Frequencies in nv_axes order (GHz) ---")
    # Using the permutation, reorder the measured lines into nv_axes order:
    f_nvorder = np.empty(4, float)
    for in_idx, nv_idx in enumerate(out["perm"]):
        f_nvorder[nv_idx] = f_any_order[in_idx]
    print(" measured f_- (nv_axes order):", np.round(f_nvorder, 12))
    print(" predicted f_- (nv_axes order):", np.round(out["f_minus_nvaxes_GHz"], 12))

    print("\n--- Fit quality ---")
    print(
        f"Residual norm: {out['residual_norm']:.3e}   |   "
        f"sign_consistent: {out['sign_consistent']}"
    )

    # Per-line mapping table in the original input order,
    # using the stored perm (no re-optimization).
    print("\n--- Frequency → Orientation mapping (original input order) ---")
    abs_nB_meas_all = (D_GHz - f_any_order) * 1000.0 / float(gamma_e_MHz_per_G)

    sq_errs = []
    for in_idx, nv_idx in enumerate(out["perm"]):
        label = NV_LABELS[nv_idx]
        abs_meas = abs_nB_meas_all[in_idx]
        abs_pred = out["abs_nB_pred"][nv_idx]
        sq_errs.append((abs_meas - abs_pred) ** 2)

        print(
            f" meas idx {in_idx}: f_-={f_any_order[in_idx]:.9f} GHz  →  NV {label} "
            f"(ori idx {nv_idx});  |n·B|_meas={abs_meas:.6f} G, "
            f"|n·B|_pred={abs_pred:.6f} G"
        )

    rmsG = float(np.sqrt(np.mean(sq_errs)))
    print(f" RMS(|n·B| mismatch) ≈ {rmsG:.6e} G")
    print("========================================================\n")


# (Optional) – If you still want a standalone "best assignment" helper
# not used in print_full_summary anymore, but handy for debugging:
def map_frequencies_to_orientations(
    f_ms_minus_GHz, B_crystal,
    D_GHz=2.8785, gamma_e_MHz_per_G=2.8025,
):
    """
    Given any-order f_- and a known B (crystal), find the permutation of
    lines -> NV axes that minimizes |n·B| mismatch. Useful for debugging
    or cross-checking, but NOT used in the locked-gauge printing pipeline.
    """
    f = np.asarray(f_ms_minus_GHz, float).reshape(4)
    abs_nB_meas = (D_GHz - f) * 1000.0 / float(gamma_e_MHz_per_G)

    n = nv_axes()
    abs_nB_pred_all = np.abs(n @ np.asarray(B_crystal, float).reshape(3))

    # Brute-force minimal assignment
    best_perm, best_err = None, np.inf
    for perm in permutations(range(4)):
        err = 0.0
        for j in range(4):
            i = perm[j]
            d = abs_nB_meas[j] - abs_nB_pred_all[i]
            err += d * d
        if err < best_err:
            best_err, best_perm = err, perm

    rms_G = float(np.sqrt(best_err / 4.0))
    mapping = []
    for j in range(4):
        i = best_perm[j]
        mapping.append({
            "idx_meas": j,
            "f_GHz": f[j],
            "ori_idx": i,
            "ori_label": NV_LABELS[i],
            "abs_nB_meas_G": abs_nB_meas[j],
            "abs_nB_pred_G": abs_nB_pred_all[i],
        })
    return mapping, best_perm, rms_G


def align_f_new_to_reference(f_ref_nvaxes, f_new_raw):
    """
    Given:
      - f_ref_nvaxes: array length-4, f_- in NV-axes order from reference
      - f_new_raw:   array length-4, f_- in *arbitrary* new input order

    Find a permutation of f_new_raw so that
        sum_i (f_new_perm[i] - f_ref_nvaxes[i])^2
    is minimized.

    Returns:
      f_new_aligned: reordered version of f_new_raw
      perm_idx:      tuple such that f_new_aligned[i] = f_new_raw[perm_idx[i]]
    """
    f_ref_nvaxes = np.asarray(f_ref_nvaxes, float).reshape(4)
    f_new_raw = np.asarray(f_new_raw, float).reshape(4)

    best_err = np.inf
    best_perm = None

    for perm in permutations(range(4)):
        f_candidate = f_new_raw[list(perm)]
        err = np.sum((f_candidate - f_ref_nvaxes)**2)
        if err < best_err:
            best_err = err
            best_perm = perm

    f_new_aligned = f_new_raw[list(best_perm)]
    return f_new_aligned, best_perm


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Reference and new datasets (same four branches, same index order)
    f_ref = [2.7666, 2.7851, 2.8222, 2.8406]
    # f_new = [2.7245, 2.7471,  2.8480, 2.8282]
    # f_ref = [2.7058, 2.7859,  2.8280, 2.8699]
    # f_new = [2.7081, 2.8083,  2.8536, 2.8251]
    # f_new = [2.7859, 2.7098,  2.8706, 2.8169]
    
    # f_ref = [2.7829, 2.771367, 2.8202, 2.810689] ### Coils off
    # f_ref = [2.771367, 2.7825,  2.810689, 2.8220] ### Coils off
    

    # For your *real* coil step, plug in the actual 4 numbers here.

    # --- Reference solve ---
    out_ref = solve_B_from_odmr_order_invariant(f_ref)
    print("=== Reference full solve (used to define perm_ref) ===")
    print_full_summary(f_ref, out_ref)
    
    perm_ref = out_ref["perm"]   # input idx -> nv_axes idx

    # --- New solve with fixed permutation ---
    out_new = solve_B_with_fixed_perm(f_new, perm_ref)
    print("=== New solve with fixed orientation perm_ref ===")
    print_full_summary(f_new, out_new)

    # --- Orientation-ordered in solver canonical convention ---
    f_ref_nv = out_ref["f_minus_nvaxes_GHz"]
    f_new_nv = out_new["f_minus_nvaxes_GHz"]

    print("=== Orientation-ordered frequencies (canonical NV axes) ===")
    for i, lab in enumerate(NV_LABELS):
        f0 = f_ref_nv[i]
        f1 = f_new_nv[i]
        df_MHz = 1000.0 * (f1 - f0)
        print(
            f"  {lab:>8}:  ref = {f0:.6f}  →  new = {f1:.6f}   "
            f"(Δf = {df_MHz:+.3f} MHz)"
        )

    # --- Lab orientation remap (tune ORI_MAP once based on calibration) ---
    ORI_MAP = [0, 1, 2, 3]  # start with identity, adjust after you compare
    LAB_LABELS = ["[1, 1, 1]", "[-1, 1, 1]", "[1, -1, 1]", "[1, 1, -1]"]

    print("=== Lab-orientation frequencies (remapped) ===")
    for lab_idx, lab in enumerate(LAB_LABELS):
        solver_idx = ORI_MAP[lab_idx]
        f0 = f_ref_nv[solver_idx]
        f1 = f_new_nv[solver_idx]
        df_MHz = 1000.0 * (f1 - f0)
        print(
            f"  {lab:>8}:  ref = {f0:.6f}  →  new = {f1:.6f}   "
            f"(Δf = {df_MHz:+.3f} MHz)"
        )

    # --- B-field comparison ---
    dB = out_new["B"] - out_ref["B"]
    print("\n=== B-field change ===")
    print("B_ref (G):", np.round(out_ref["B"], 6))
    print("B_new (G):", np.round(out_new["B"], 6))
    print("ΔB (G)   :", np.round(dB, 6))

