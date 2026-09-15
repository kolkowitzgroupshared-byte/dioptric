# -*- coding: utf-8 -*-
"""
Robust 8-line NV ODMR magnetic-field solver.

Key differences from the older 4-line solver
---------------------------------------------
1. Uses all 8 ODMR lines.
2. Does NOT assume the lower/upper pairing is correct:
   all 4! = 24 lower<->upper pairings are tested.
3. Uses the full spin-1 Hamiltonian:
       H/h = D Sz^2 + E(Sx^2 - Sy^2) + gamma_e B.S
4. Does NOT use an old field vector to determine signs or axis labels.
5. Explicitly reports the cubic-symmetry ambiguity.
6. Prints model-consistent ("refined") peak positions separately from the
   manually picked measured peaks.
7. An old field can optionally be used ONLY as a continuity/ranking hint
   among already-degenerate solutions. It never changes the spectral fit.

For a spectrum alone, robust quantities are:
    - |B|
    - the unordered absolute cubic-component magnitudes, modulo symmetry
    - the best lower<->upper pairing
    - D (within the assumed Hamiltonian model)

Absolute lab-frame signs / axis labels require an independent calibration.

Author: Saroj Chand
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import numpy as np
from scipy.optimize import least_squares


# =============================================================================
# USER SETTINGS
# =============================================================================

# Eight manually picked ESR peak centers, GHz.
ODMR_PEAKS_GHZ = np.array(
    [
        2.7539,
        2.7773,
        2.8212,
        2.8421,
        2.9195,
        2.9370,
        2.9758,
        2.9906,
    ],
    dtype=float,
)

# Electron gyromagnetic ratio.
GAMMA_E_GHZ_PER_G = 2.8025e-3

# Keep E = 0 unless you have an independently calibrated transverse strain
# term and a consistent local x/y convention for all NV orientations.
E_GHZ = 0.0

# Fit bounds.
B_COMPONENT_BOUND_G = 300.0
D_BOUNDS_GHZ = (2.84, 2.90)

# Nonlinear optimizer.
MAX_NFEV = 5000

# Near-best spectral solutions within this RMS difference are considered
# spectroscopically degenerate. 1e-4 MHz is intentionally very tight.
NEAR_BEST_RMS_MHZ = 1e-4

# Number of pairing candidates and symmetry solutions to print.
TOP_PAIRINGS_TO_PRINT = 10
MAX_SYMMETRY_SOLUTIONS_TO_PRINT = 48

# ---------------------------------------------------------------------------
# OPTIONAL continuity hint
# ---------------------------------------------------------------------------
# Your previous vector may itself have come from a symmetry-ambiguous ODMR
# solve. Therefore the default is FALSE.
#
# If, later, you know that the previous field is physically calibrated in the
# SAME crystal coordinate frame, change this to True. Then it is used only to
# choose/rank one branch among already spectrally equivalent solutions.
USE_PREVIOUS_FIELD_HINT = False

PREVIOUS_B_G = np.array(
    [-46.275577, -17.165999, -5.701398],
    dtype=float,
)


# =============================================================================
# NV GEOMETRY
# =============================================================================

def nv_axes() -> np.ndarray:
    """
    Four unoriented NV <111> axes in cubic coordinates:
        x = [100], y = [010], z = [001]

    The sign chosen for each individual NV axis is a convention; an NV axis
    physically represents a line. This set is equivalent to the tetrahedral
    family and satisfies sum_i n_i n_i^T = (4/3) I.
    """
    axes = np.array(
        [
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes


NV_AXES = nv_axes()

NV_LABELS = (
    "[ 1, 1, 1]",
    "[-1, 1, 1]",
    "[ 1,-1, 1]",
    "[ 1, 1,-1]",
)


# =============================================================================
# SPIN-1 MATRICES
# =============================================================================

SQRT2 = np.sqrt(2.0)

SX = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=complex,
) / SQRT2

SY = np.array(
    [
        [0.0, -1.0j, 0.0],
        [1.0j, 0.0, -1.0j],
        [0.0, 1.0j, 0.0],
    ],
    dtype=complex,
) / SQRT2

SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)
SZ2 = SZ @ SZ


# =============================================================================
# LOCAL FRAME FOR EACH NV
# =============================================================================

def local_basis_from_nv_axis(nv_axis: np.ndarray):
    """
    Build a right-handed local x',y',z' frame with z' along the NV axis.

    For E=0, rotation of x'/y' around z' does not affect the eigenvalues.
    """
    z_local = np.asarray(nv_axis, dtype=float)
    z_local /= np.linalg.norm(z_local)

    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(ref, z_local)) > 0.90:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    x_local = np.cross(ref, z_local)
    x_local /= np.linalg.norm(x_local)

    y_local = np.cross(z_local, x_local)
    y_local /= np.linalg.norm(y_local)

    return x_local, y_local, z_local


LOCAL_BASES = [
    local_basis_from_nv_axis(axis)
    for axis in NV_AXES
]


# =============================================================================
# HAMILTONIAN
# =============================================================================

def transition_frequencies_for_axis(
    b_crystal_g: np.ndarray,
    d_ghz: float,
    axis_index: int,
    e_ghz: float = E_GHZ,
) -> np.ndarray:
    """
    Calculate the two ODMR transition frequencies for one NV axis.

    Hamiltonian in frequency units (GHz):
        H/h = D Sz^2 + E(Sx^2-Sy^2) + gamma_e (Bx Sx + By Sy + Bz Sz)

    At the fields considered here, the lowest eigenstate remains the
    ms=0-like state. The two measured transitions are E1-E0 and E2-E0.
    """
    b = np.asarray(b_crystal_g, dtype=float).reshape(3)

    x_local, y_local, z_local = LOCAL_BASES[axis_index]

    b_local = np.array(
        [
            np.dot(b, x_local),
            np.dot(b, y_local),
            np.dot(b, z_local),
        ],
        dtype=float,
    )

    h = (
        d_ghz * SZ2
        + e_ghz * (SX @ SX - SY @ SY)
        + GAMMA_E_GHZ_PER_G
        * (
            b_local[0] * SX
            + b_local[1] * SY
            + b_local[2] * SZ
        )
    )

    evals = np.linalg.eigvalsh(h).real
    evals.sort()

    transitions = np.array(
        [
            evals[1] - evals[0],
            evals[2] - evals[0],
        ],
        dtype=float,
    )
    transitions.sort()

    return transitions


def predict_all_axes(
    b_crystal_g: np.ndarray,
    d_ghz: float,
) -> np.ndarray:
    """Shape (4,2): low/high ODMR transition for each NV axis."""
    return np.vstack(
        [
            transition_frequencies_for_axis(
                b_crystal_g,
                d_ghz,
                axis_index=i,
            )
            for i in range(4)
        ]
    )


# =============================================================================
# PEAK PARTITION AND ALL POSSIBLE LOWER<->UPPER PAIRINGS
# =============================================================================

def split_low_high(peaks_ghz: np.ndarray):
    """
    Sort 8 peaks and take the lower four and upper four.

    This assumes the field is modest enough that all four lower branches lie
    below all four upper branches, which is true for the present spectrum.
    """
    f = np.sort(
        np.asarray(peaks_ghz, dtype=float).reshape(8)
    )
    lower = f[:4].copy()
    upper = f[4:].copy()
    return lower, upper


def all_pairings(peaks_ghz: np.ndarray):
    """
    Generate all 24 ways to pair lower lines with upper lines.

    Pair index is tied to the sorted lower-line index:
        pairs[i] = [lower[i], selected upper line]
    """
    lower, upper = split_low_high(peaks_ghz)

    candidates = []

    for upper_perm in permutations(range(4)):
        pairs = np.column_stack(
            (
                lower,
                upper[list(upper_perm)],
            )
        )
        candidates.append(
            (
                tuple(upper_perm),
                pairs,
            )
        )

    return candidates


# =============================================================================
# FIRST-ORDER ESTIMATES USED ONLY AS ROBUST INITIAL GUESSES / CROSS-CHECKS
# =============================================================================

def first_order_from_pairs(
    pairs_ghz: np.ndarray,
):
    """
    Use BOTH lines of each pair:

        |B_parallel| = (f_high-f_low)/(2 gamma_e)

    and tetrahedral identity:

        |B|^2 = (3/4) sum_i |B_parallel,i|^2
    """
    pairs = np.asarray(
        pairs_ghz,
        dtype=float,
    ).reshape(4, 2)

    splitting_ghz = (
        pairs[:, 1] - pairs[:, 0]
    )

    abs_b_parallel_g = (
        splitting_ghz
        / (2.0 * GAMMA_E_GHZ_PER_G)
    )

    b_mag_g = np.sqrt(
        0.75
        * np.sum(abs_b_parallel_g**2)
    )

    centers_ghz = 0.5 * (
        pairs[:, 0] + pairs[:, 1]
    )

    return {
        "splitting_ghz": splitting_ghz,
        "abs_b_parallel_g": abs_b_parallel_g,
        "b_mag_g": float(b_mag_g),
        "centers_ghz": centers_ghz,
    }


def unique_sign_patterns():
    """
    B and -B produce the same unordered ODMR transition pairs.
    Therefore fix the first projection sign positive and search 2^3=8
    relative-sign patterns; the opposite B branch is generated automatically.
    """
    return [
        (1, s1, s2, s3)
        for s1, s2, s3 in product(
            [-1, +1],
            repeat=3,
        )
    ]


def linear_b_seed(
    abs_b_parallel_pair_order_g: np.ndarray,
    axis_to_pair: tuple[int, int, int, int],
    signs: tuple[int, int, int, int],
):
    """
    Construct a first-order B seed.

    IMPORTANT convention used everywhere:
        axis_to_pair[axis_idx] = measured_pair_idx
    """
    bmag_axis_order = np.asarray(
        abs_b_parallel_pair_order_g,
        dtype=float,
    )[list(axis_to_pair)]

    rhs = (
        np.asarray(signs, dtype=float)
        * bmag_axis_order
    )

    b_seed, *_ = np.linalg.lstsq(
        NV_AXES,
        rhs,
        rcond=None,
    )

    return b_seed


# =============================================================================
# NONLINEAR FIT
# =============================================================================

@dataclass
class FitSolution:
    rms_mhz: float
    max_abs_residual_mhz: float

    b_g: np.ndarray
    b_mag_g: float
    b_hat: np.ndarray

    d_ghz: float

    # axis_to_pair[axis_idx] = measured_pair_idx
    axis_to_pair: tuple[int, int, int, int]

    measured_pairs_axis_order_ghz: np.ndarray
    predicted_pairs_axis_order_ghz: np.ndarray
    residuals_axis_order_mhz: np.ndarray

    success: bool
    nfev: int


def fit_assignment(
    pairs_ghz: np.ndarray,
    axis_to_pair: tuple[int, int, int, int],
    b_seed_g: np.ndarray,
    d_seed_ghz: float,
):
    """
    Fit Bx, By, Bz and D for a fixed pair->axis assignment.
    """
    pairs = np.asarray(
        pairs_ghz,
        dtype=float,
    ).reshape(4, 2)

    measured_axis_order = pairs[
        list(axis_to_pair)
    ]

    x0 = np.array(
        [
            b_seed_g[0],
            b_seed_g[1],
            b_seed_g[2],
            d_seed_ghz,
        ],
        dtype=float,
    )

    lower = np.array(
        [
            -B_COMPONENT_BOUND_G,
            -B_COMPONENT_BOUND_G,
            -B_COMPONENT_BOUND_G,
            D_BOUNDS_GHZ[0],
        ],
        dtype=float,
    )

    upper = np.array(
        [
            B_COMPONENT_BOUND_G,
            B_COMPONENT_BOUND_G,
            B_COMPONENT_BOUND_G,
            D_BOUNDS_GHZ[1],
        ],
        dtype=float,
    )

    x0 = np.clip(
        x0,
        lower + 1e-12,
        upper - 1e-12,
    )

    def residual_vector(x):
        pred = predict_all_axes(
            x[:3],
            x[3],
        )

        # MHz residuals.
        return (
            (pred - measured_axis_order)
            * 1000.0
        ).ravel()

    res = least_squares(
        residual_vector,
        x0,
        bounds=(lower, upper),
        max_nfev=MAX_NFEV,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        method="trf",
    )

    b = np.asarray(
        res.x[:3],
        dtype=float,
    )
    d = float(res.x[3])

    pred = predict_all_axes(
        b,
        d,
    )

    residuals_mhz = (
        pred - measured_axis_order
    ) * 1000.0

    flat = residuals_mhz.ravel()

    rms_mhz = float(
        np.sqrt(
            np.mean(flat**2)
        )
    )

    max_abs_mhz = float(
        np.max(np.abs(flat))
    )

    b_mag = float(
        np.linalg.norm(b)
    )

    b_hat = (
        b / b_mag
        if b_mag > 0
        else np.zeros(3, dtype=float)
    )

    return FitSolution(
        rms_mhz=rms_mhz,
        max_abs_residual_mhz=max_abs_mhz,
        b_g=b,
        b_mag_g=b_mag,
        b_hat=b_hat,
        d_ghz=d,
        axis_to_pair=tuple(axis_to_pair),
        measured_pairs_axis_order_ghz=measured_axis_order,
        predicted_pairs_axis_order_ghz=pred,
        residuals_axis_order_mhz=residuals_mhz,
        success=bool(res.success),
        nfev=int(res.nfev),
    )


def fit_pairing_spectrally(
    pairs_ghz: np.ndarray,
):
    """
    Score ONE lower<->upper pairing.

    Due cubic symmetry, the minimum spectral residual for a given pairing
    does not require searching all 24 axis labels. We use identity pair->axis
    assignment here and search all relative projection signs.

    This stage determines the PHYSICAL LOWER<->UPPER PAIRING independently
    of any old field vector.
    """
    cross = first_order_from_pairs(
        pairs_ghz
    )

    d_seed = float(
        np.median(
            cross["centers_ghz"]
        )
    )

    axis_to_pair = (0, 1, 2, 3)

    sols = []

    for signs in unique_sign_patterns():
        b_seed = linear_b_seed(
            cross["abs_b_parallel_g"],
            axis_to_pair,
            signs,
        )

        sol = fit_assignment(
            pairs_ghz,
            axis_to_pair,
            b_seed,
            d_seed,
        )

        if (
            sol.success
            and np.isfinite(sol.rms_mhz)
        ):
            sols.append(sol)

    if not sols:
        return None

    sols.sort(
        key=lambda s: (
            s.rms_mhz,
            s.max_abs_residual_mhz,
        )
    )

    return sols[0]


def search_all_peak_pairings(
    peaks_ghz: np.ndarray,
):
    """
    Search all 24 lower<->upper pairings and return ranked results.
    """
    results = []

    for upper_perm, pairs in all_pairings(
        peaks_ghz
    ):
        sol = fit_pairing_spectrally(
            pairs
        )

        if sol is not None:
            results.append(
                {
                    "upper_perm": upper_perm,
                    "pairs": pairs,
                    "solution": sol,
                }
            )

    if not results:
        raise RuntimeError(
            "No lower<->upper pairing fit converged."
        )

    results.sort(
        key=lambda item: (
            item["solution"].rms_mhz,
            item["solution"].max_abs_residual_mhz,
        )
    )

    return results


def solve_all_crystal_symmetry_branches(
    best_pairs_ghz: np.ndarray,
):
    """
    After choosing the best lower<->upper pairing, search all 24 mappings
    of those four pairs onto the four NV axes.

    All symmetry-related solutions should have essentially the same residual.
    """
    pairs = np.asarray(
        best_pairs_ghz,
        dtype=float,
    ).reshape(4, 2)

    cross = first_order_from_pairs(
        pairs
    )

    d_seed = float(
        np.median(
            cross["centers_ghz"]
        )
    )

    solutions = []

    for axis_to_pair in permutations(
        range(4)
    ):
        for signs in unique_sign_patterns():
            b_seed = linear_b_seed(
                cross["abs_b_parallel_g"],
                axis_to_pair,
                signs,
            )

            sol = fit_assignment(
                pairs,
                tuple(axis_to_pair),
                b_seed,
                d_seed,
            )

            if (
                sol.success
                and np.isfinite(sol.rms_mhz)
            ):
                solutions.append(sol)

                # Explicitly add the B -> -B partner if it is not already
                # generated by numerical convergence.
                sol_minus = fit_assignment(
                    pairs,
                    tuple(axis_to_pair),
                    -sol.b_g,
                    sol.d_ghz,
                )
                if (
                    sol_minus.success
                    and np.isfinite(
                        sol_minus.rms_mhz
                    )
                ):
                    solutions.append(
                        sol_minus
                    )

    if not solutions:
        raise RuntimeError(
            "No symmetry-branch fit converged."
        )

    solutions.sort(
        key=lambda s: (
            s.rms_mhz,
            s.max_abs_residual_mhz,
        )
    )

    return solutions


# =============================================================================
# DEDUPLICATION AND OPTIONAL CONTINUITY RANKING
# =============================================================================

def same_solution(
    a: FitSolution,
    b: FitSolution,
    atol_b_g=1e-5,
    atol_d_ghz=1e-10,
):
    return bool(
        np.allclose(
            a.b_g,
            b.b_g,
            atol=atol_b_g,
            rtol=0.0,
        )
        and abs(
            a.d_ghz - b.d_ghz
        )
        <= atol_d_ghz
    )


def unique_near_best_solutions(
    solutions,
):
    best_rms = solutions[0].rms_mhz

    near = [
        s
        for s in solutions
        if s.rms_mhz
        <= best_rms
        + NEAR_BEST_RMS_MHZ
    ]

    unique = []

    for sol in near:
        if not any(
            same_solution(
                sol,
                old,
            )
            for old in unique
        ):
            unique.append(sol)

    return unique


def angle_deg(
    a: np.ndarray,
    b: np.ndarray,
):
    a = np.asarray(
        a,
        dtype=float,
    )
    b = np.asarray(
        b,
        dtype=float,
    )

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na == 0 or nb == 0:
        return np.nan

    c = np.dot(a, b) / (na * nb)
    c = np.clip(c, -1.0, 1.0)

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


def choose_display_solution(
    near_best_solutions,
):
    """
    Choose one representative for printing.

    Without an independently calibrated old field:
        choose a deterministic canonical representative only.

    With USE_PREVIOUS_FIELD_HINT=True:
        choose the spectrally near-degenerate branch closest in angle to
        PREVIOUS_B_G. This is ONLY continuity ranking, not a spectral
        determination of the signs.
    """
    sols = list(
        near_best_solutions
    )

    if USE_PREVIOUS_FIELD_HINT:
        sols.sort(
            key=lambda s: (
                angle_deg(
                    s.b_g,
                    PREVIOUS_B_G,
                ),
                s.rms_mhz,
            )
        )
        return sols[0]

    # Deterministic canonical choice.
    sols.sort(
        key=lambda s: tuple(
            np.round(
                s.b_g,
                9,
            ).tolist()
        )
    )

    return sols[0]


# =============================================================================
# MODEL-CONSISTENT PEAK POSITIONS
# =============================================================================

def predicted_pairs_in_measured_pair_order(
    sol: FitSolution,
):
    """
    Convert predicted axis-order pairs back to measured-pair order.
    """
    pred_pair_order = np.empty(
        (4, 2),
        dtype=float,
    )

    for axis_idx, pair_idx in enumerate(
        sol.axis_to_pair
    ):
        pred_pair_order[pair_idx] = (
            sol.predicted_pairs_axis_order_ghz[
                axis_idx
            ]
        )

    return pred_pair_order


# =============================================================================
# PRINTING
# =============================================================================

def print_pairing_search(
    pairing_results,
):
    print()
    print(
        "=============================================================="
    )
    print(
        "LOWER <-> UPPER PAIRING SEARCH"
    )
    print(
        "=============================================================="
    )
    print(
        "No previous B vector is used here."
    )
    print()

    print(
        "rank   RMS(MHz)   max|res|(MHz)   upper permutation   pairs (GHz)"
    )
    print(
        "--------------------------------------------------------------------------"
    )

    for rank, item in enumerate(
        pairing_results[
            :TOP_PAIRINGS_TO_PRINT
        ],
        start=1,
    ):
        sol = item["solution"]

        compact_pairs = "; ".join(
            f"{lo:.4f}<->{hi:.4f}"
            for lo, hi in item["pairs"]
        )

        print(
            f"{rank:>3d}    "
            f"{sol.rms_mhz:>9.4f}      "
            f"{sol.max_abs_residual_mhz:>9.4f}       "
            f"{item['upper_perm']}     "
            f"{compact_pairs}"
        )


def print_first_order(
    pairs,
):
    out = first_order_from_pairs(
        pairs
    )

    print()
    print(
        "=============================================================="
    )
    print(
        "FIRST-ORDER CROSS-CHECK FOR BEST PAIRING"
    )
    print(
        "=============================================================="
    )

    print(
        "pair    low(GHz)    high(GHz)    split(MHz)    |Bpar|(G)    center(GHz)"
    )
    print(
        "-----------------------------------------------------------------------"
    )

    for i in range(4):
        print(
            f"{i:>3d}    "
            f"{pairs[i,0]:>9.6f}   "
            f"{pairs[i,1]:>9.6f}   "
            f"{1000*out['splitting_ghz'][i]:>10.3f}   "
            f"{out['abs_b_parallel_g'][i]:>9.4f}   "
            f"{out['centers_ghz'][i]:>11.6f}"
        )

    print()
    print(
        f"First-order |B| = "
        f"{out['b_mag_g']:.6f} G "
        f"= {out['b_mag_g']/10:.6f} mT"
    )


def print_solution(
    sol: FitSolution,
):
    print()
    print(
        "=============================================================="
    )
    print(
        "FULL 8-LINE HAMILTONIAN FIT — ONE REPRESENTATIVE"
    )
    print(
        "=============================================================="
    )

    print(
        "IMPORTANT: unless independently calibrated, this Bx/By/Bz branch"
    )
    print(
        "is only one symmetry-equivalent crystal-coordinate representation."
    )
    print()

    print(
        f"B (G) = {np.array2string(sol.b_g, precision=9)}"
    )
    print(
        f"|B|   = {sol.b_mag_g:.9f} G "
        f"= {sol.b_mag_g/10:.9f} mT"
    )
    print(
        f"B_hat = {np.array2string(sol.b_hat, precision=9)}"
    )
    print(
        f"D      = {sol.d_ghz:.9f} GHz"
    )
    print(
        f"E      = {E_GHZ:.9f} GHz (fixed)"
    )
    print(
        f"RMS residual = {sol.rms_mhz:.6f} MHz"
    )
    print(
        f"Max residual = "
        f"{sol.max_abs_residual_mhz:.6f} MHz"
    )

    abs_sorted = np.sort(
        np.abs(sol.b_g)
    )[::-1]

    print()
    print(
        "Robust component magnitudes modulo cubic symmetry:"
    )
    print(
        f"sorted(|Bx|,|By|,|Bz|) = "
        f"{np.array2string(abs_sorted, precision=6)} G"
    )


def print_refined_peaks(
    measured_pairs,
    sol,
):
    pred_pairs = (
        predicted_pairs_in_measured_pair_order(
            sol
        )
    )

    print()
    print(
        "=============================================================="
    )
    print(
        "MANUAL PEAKS VS MODEL-CONSISTENT PEAKS"
    )
    print(
        "=============================================================="
    )
    print(
        "The predicted values below are NOT replacement measurements."
    )
    print(
        "They are the nearest peak positions constrained by the fitted model."
    )
    print()

    print(
        "pair    measured low   model low    dlow(MHz)   "
        "measured high  model high   dhigh(MHz)"
    )
    print(
        "----------------------------------------------------------------------------"
    )

    for i in range(4):
        m = measured_pairs[i]
        p = pred_pairs[i]
        delta = (
            p - m
        ) * 1000.0

        print(
            f"{i:>3d}     "
            f"{m[0]:.6f}     "
            f"{p[0]:.6f}    "
            f"{delta[0]:+8.3f}      "
            f"{m[1]:.6f}     "
            f"{p[1]:.6f}    "
            f"{delta[1]:+8.3f}"
        )

    measured_flat = np.sort(
        measured_pairs.ravel()
    )

    model_flat = np.sort(
        pred_pairs.ravel()
    )

    print()
    print(
        "Measured 8 peaks:"
    )
    print(
        np.array2string(
            measured_flat,
            precision=6,
            separator=", ",
        )
    )

    print(
        "Model-consistent 8 peaks:"
    )
    print(
        np.array2string(
            model_flat,
            precision=6,
            separator=", ",
        )
    )


def print_symmetry_solutions(
    unique_solutions,
):
    print()
    print(
        "=============================================================="
    )
    print(
        "SPECTROSCOPICALLY EQUIVALENT CRYSTAL-FRAME BRANCHES"
    )
    print(
        "=============================================================="
    )

    print(
        f"Unique near-best B vectors: "
        f"{len(unique_solutions)}"
    )
    print()

    if USE_PREVIOUS_FIELD_HINT:
        print(
            "Previous field is being used ONLY as a continuity ranking hint:"
        )
        print(
            f"  PREVIOUS_B_G = {PREVIOUS_B_G}"
        )
        print(
            "This does not make that old branch independently calibrated."
        )
        print()

    print(
        " #        Bx(G)        By(G)        Bz(G)      "
        "|B|(G)     RMS(MHz)"
        + (
            "    angle-to-old(deg)"
            if USE_PREVIOUS_FIELD_HINT
            else ""
        )
    )
    print(
        "--------------------------------------------------------------------------"
    )

    sols = list(
        unique_solutions
    )

    if USE_PREVIOUS_FIELD_HINT:
        sols.sort(
            key=lambda s: angle_deg(
                s.b_g,
                PREVIOUS_B_G,
            )
        )

    for i, sol in enumerate(
        sols[
            :MAX_SYMMETRY_SOLUTIONS_TO_PRINT
        ]
    ):
        line = (
            f"{i:2d}   "
            f"{sol.b_g[0]:11.6f} "
            f"{sol.b_g[1]:11.6f} "
            f"{sol.b_g[2]:11.6f} "
            f"{sol.b_mag_g:10.6f} "
            f"{sol.rms_mhz:10.6f}"
        )

        if USE_PREVIOUS_FIELD_HINT:
            line += (
                f"      "
                f"{angle_deg(sol.b_g, PREVIOUS_B_G):10.4f}"
            )

        print(line)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print(
        "Input peaks (GHz):"
    )
    print(
        np.sort(
            ODMR_PEAKS_GHZ
        )
    )

    # -------------------------------------------------------------------------
    # 1. Search all lower<->upper pairings using ONLY the current spectrum.
    # -------------------------------------------------------------------------
    pairing_results = (
        search_all_peak_pairings(
            ODMR_PEAKS_GHZ
        )
    )

    print_pairing_search(
        pairing_results
    )

    best_pairs = pairing_results[0][
        "pairs"
    ]

    print()
    print(
        "Best lower<->upper pairing:"
    )
    print(
        best_pairs
    )

    # -------------------------------------------------------------------------
    # 2. First-order magnitude cross-check.
    # -------------------------------------------------------------------------
    print_first_order(
        best_pairs
    )

    # -------------------------------------------------------------------------
    # 3. Search all crystallographic symmetry branches for best pairing.
    # -------------------------------------------------------------------------
    print()
    print(
        "Searching all crystallographic symmetry branches ..."
    )

    all_solutions = (
        solve_all_crystal_symmetry_branches(
            best_pairs
        )
    )

    unique_near_best = (
        unique_near_best_solutions(
            all_solutions
        )
    )

    display_solution = (
        choose_display_solution(
            unique_near_best
        )
    )

    # -------------------------------------------------------------------------
    # 4. Report one representative plus invariant quantities.
    # -------------------------------------------------------------------------
    print_solution(
        display_solution
    )

    # -------------------------------------------------------------------------
    # 5. Show how much the manually chosen peak centers would move if forced
    #    onto the fitted spin-Hamiltonian manifold.
    # -------------------------------------------------------------------------
    print_refined_peaks(
        best_pairs,
        display_solution,
    )

    # -------------------------------------------------------------------------
    # 6. Explicitly show all spectroscopically equivalent branches.
    # -------------------------------------------------------------------------
    print_symmetry_solutions(
        unique_near_best
    )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------
    print()
    print(
        "=============================================================="
    )
    print(
        "INTERPRETATION"
    )
    print(
        "=============================================================="
    )
    print(
        "Trust |B| and the unordered absolute component magnitudes from this"
    )
    print(
        "spectrum. Do NOT interpret the printed Cartesian signs or axis labels"
    )
    print(
        "as physical lab-frame direction until an independent orientation"
    )
    print(
        "calibration is available (known coil step, crystal-edge mapping, etc.)."
    )
