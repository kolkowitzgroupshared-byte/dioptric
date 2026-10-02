# -*- coding: utf-8 -*-
"""
Widefield DEER Hahn analysis

Dataset:
    2026_09_24-17_24_18-qnami-nv0_2026_02_20

Assumed acquisition ordering:
    [f0_ON, f0_OFF, f1_ON, f1_OFF, ...]

where OFF is the detuned-RF reference used during acquisition.

Outputs are saved through utils.data_manager.get_file_path(),
so they follow the same standard data hierarchy used by the experiment
(e.g. G:\\nvdata on Purcell).

Outputs:
    1. Multi-page PDF
       - all NV spectra
       - median + IQR
       - DEER heatmap
       - strongest-response NVs
       - all individual NV spectra
    2. CSV ranking of NV responses
    3. NPZ with processed arrays
"""

import csv
import io
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import eigh

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield


# FILE_STEM = [
#             "2026_09_24-17_24_18-qnami-nv0_2026_02_20",
#             "2026_09_24-19_57_56-qnami-nv0_2026_02_20",
#             "2026_09_24-22_45_13-qnami-nv0_2026_02_20",
#             "2026_09_25-02_45_17-qnami-nv0_2026_02_20",
#             "2026_09_25-06_04_42-qnami-nv0_2026_02_20",
#             "2026_09_25-09_28_49-qnami-nv0_2026_02_20",
#             ]

# FILE_STEM = ["2026_09_26-02_01_14-qnami-nv0_2026_02_20"]

# FILE_STEM = ["2026_09_26-15_19_45-qnami-nv0_2026_02_20",
#              "2026_09_26-18_04_56-qnami-nv0_2026_02_20",
#              "2026_09_26-20_49_00-qnami-nv0_2026_02_20",
#              ]

FILE_STEM = [
             "2026_09_28-18_00_37-qnami-nv0_2026_02_20",
             "2026_09_29-00_44_53-qnami-nv0_2026_02_20",
             "2026_09_29-07_14_51-qnami-nv0_2026_02_20",
             "2026_09_29-14_10_23-qnami-nv0_2026_02_20",
             ]
DYNAMIC_THRESHOLD = False
NV_PER_PAGE = 12
TOP_N = 20

# 52 G orientation map from resonance/spin-echo work:
# group_A = orientation (1, 1, -1)
# low-frequency ESR branch ~2.7773 GHz
# high-frequency ESR branch ~2.9758 GHz
# These are CURRENT indices in the 212-NV qnami list.
TARGET_ESR_GHZ = 2.7773
TARGET_ORIENTATION = "(1, 1, -1)"
TARGET_GROUP = "group_A"

TARGET_NV_INDICES = np.array(
    [0, 8, 9, 11, 12, 14, 15, 21, 22, 24, 26, 27, 29, 31, 32, 33, 35, 39, 41, 44, 46, 48, 50, 53, 57, 61, 64, 66, 67, 70, 72, 73, 75, 79, 81, 82, 83, 84, 86, 87, 88, 90, 91, 92, 93, 95, 97, 98, 102, 106, 108, 110, 111, 114, 116, 117, 118, 119, 125, 126, 129, 132, 133, 134, 136, 139, 140, 141, 142, 146, 151, 156, 157, 159, 160, 162, 163, 166, 168, 170, 171, 173, 174, 175, 177, 178, 179, 182, 184, 185, 192, 193, 196, 197, 198, 199, 200, 202, 204, 206, 207, 208],
    dtype=int,
)

# ============================================================
# P1 simulation overlay for the current QNami field
# ============================================================

B_VEC = np.array([-48.55, -18.75, -5.97], dtype=float)  # G

GAMMA_E = 2.8025       # MHz/G
GAMMA_N14 = 3.077e-4   # MHz/G
A_PAR = 114.0           # MHz
A_PERP = 81.3           # MHz
P_PAR = -3.97           # MHz

# Bath pi-pulse duration used only for the bandwidth annotation.
RF_PI_NS_FALLBACK = 100.0

P1_WEIGHT_THRESHOLD = 0.005
P1_FULL_RANGE_MHZ = (10.0, 300.0)

JT_AXES = np.array(
    [
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ],
    dtype=float,
) / np.sqrt(3)

JT_LABELS = [
    "JT-A [111]",
    "JT-B [-1-11]",
    "JT-C [-11-1]",
    "JT-D [1-1-1]",
]

# Consistent colorblind-friendly colors for the four P1/JT families.
JT_COLORS = [
    "#D55E00",  # JT-A: vermillion
    "#0072B2",  # JT-B: blue
    "#009E73",  # JT-C: green
    "#CC79A7",  # JT-D: purple
]

def build_p1_spin_operators():
    """P1 operators for S=1/2 x I=1."""
    sx_half = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy_half = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz_half = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    eye2 = np.eye(2, dtype=complex)

    sq2 = np.sqrt(2)
    ix_one = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        dtype=complex,
    ) / sq2
    iy_one = np.array(
        [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]],
        dtype=complex,
    ) / sq2
    iz_one = np.diag([1.0, 0.0, -1.0]).astype(complex)
    eye3 = np.eye(3, dtype=complex)

    S_ops = [
        np.kron(sx_half, eye3),
        np.kron(sy_half, eye3),
        np.kron(sz_half, eye3),
    ]
    I_ops = [
        np.kron(eye2, ix_one),
        np.kron(eye2, iy_one),
        np.kron(eye2, iz_one),
    ]
    return S_ops, I_ops


def build_p1_hamiltonian(B_vec, jt_axis, S_ops, I_ops):
    """Full 6x6 P1 Hamiltonian in MHz."""
    Sx, Sy, Sz = S_ops
    Ix, Iy, Iz = I_ops
    n = jt_axis

    H_ez = GAMMA_E * (
        B_vec[0] * Sx
        + B_vec[1] * Sy
        + B_vec[2] * Sz
    )
    H_nz = -GAMMA_N14 * (
        B_vec[0] * Ix
        + B_vec[1] * Iy
        + B_vec[2] * Iz
    )

    S_dot_I = Sx @ Ix + Sy @ Iy + Sz @ Iz
    n_dot_S = n[0] * Sx + n[1] * Sy + n[2] * Sz
    n_dot_I = n[0] * Ix + n[1] * Iy + n[2] * Iz

    H_hf = (
        A_PERP * S_dot_I
        + (A_PAR - A_PERP) * (n_dot_S @ n_dot_I)
    )
    H_q = P_PAR * (
        n_dot_I @ n_dot_I
        - (2.0 / 3.0) * np.eye(6, dtype=complex)
    )

    return H_ez + H_nz + H_hf + H_q


def compute_p1_transitions(B_vec=B_VEC, intensity_threshold=1e-4):
    """
    P1 transitions for all four JT axes.

    Relative DEER weight:
        transverse MW matrix element x Delta<S_B>,
    where S_B is electron spin along the static field.
    """
    S_ops, I_ops = build_p1_spin_operators()
    Sx, Sy, Sz = S_ops

    B_hat = B_vec / np.linalg.norm(B_vec)

    if abs(B_hat[2]) < 0.9:
        perp1 = np.cross(B_hat, [0, 0, 1])
    else:
        perp1 = np.cross(B_hat, [1, 0, 0])

    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(B_hat, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    S_perp1 = (
        perp1[0] * Sx
        + perp1[1] * Sy
        + perp1[2] * Sz
    )
    S_perp2 = (
        perp2[0] * Sx
        + perp2[1] * Sy
        + perp2[2] * Sz
    )
    S_B = (
        B_hat[0] * Sx
        + B_hat[1] * Sy
        + B_hat[2] * Sz
    )

    transitions = []

    for jt_ind, (jt_axis, jt_label) in enumerate(
        zip(JT_AXES, JT_LABELS)
    ):
        H = build_p1_hamiltonian(
            B_vec,
            jt_axis,
            S_ops,
            I_ops,
        )
        evals, evecs = eigh(H)

        for i in range(6):
            for j in range(i + 1, 6):
                freq_mhz = float(evals[j] - evals[i])

                M1 = abs(
                    evecs[:, j].conj()
                    @ S_perp1
                    @ evecs[:, i]
                ) ** 2
                M2 = abs(
                    evecs[:, j].conj()
                    @ S_perp2
                    @ evecs[:, i]
                ) ** 2
                intensity = float(np.real(M1 + M2))

                if intensity <= intensity_threshold:
                    continue

                SB_i = float(
                    np.real(
                        evecs[:, i].conj()
                        @ S_B
                        @ evecs[:, i]
                    )
                )
                SB_j = float(
                    np.real(
                        evecs[:, j].conj()
                        @ S_B
                        @ evecs[:, j]
                    )
                )

                delta_SB = abs(SB_j - SB_i)
                deer_weight = intensity * delta_SB

                transitions.append(
                    {
                        "jt_ind": jt_ind,
                        "jt_label": jt_label,
                        "freq_mhz": freq_mhz,
                        "intensity": intensity,
                        "delta_SB": delta_SB,
                        "deer_weight": deer_weight,
                        "states": (i, j),
                    }
                )

    return transitions


def get_significant_p1_transitions(
    freq_min_mhz=None,
    freq_max_mhz=None,
):
    """Significant predicted P1 lines, optionally restricted to a window."""
    transitions = [
        t
        for t in compute_p1_transitions()
        if t["deer_weight"] >= P1_WEIGHT_THRESHOLD
    ]

    if freq_min_mhz is not None:
        transitions = [
            t
            for t in transitions
            if t["freq_mhz"] >= freq_min_mhz
        ]

    if freq_max_mhz is not None:
        transitions = [
            t
            for t in transitions
            if t["freq_mhz"] <= freq_max_mhz
        ]

    return sorted(
        transitions,
        key=lambda t: t["freq_mhz"],
    )


def overlay_p1_lines(
    ax,
    p1_transitions,
    label_once=True,
    alpha=0.45,
):
    """Overlay predicted P1 transitions using a consistent JT color code."""
    seen_jt = set()

    for trans in p1_transitions:
        jt_ind = int(trans["jt_ind"])
        jt_label = trans["jt_label"]
        label = None

        if label_once and jt_label not in seen_jt:
            label = jt_label
            seen_jt.add(jt_label)

        ax.axvline(
            trans["freq_mhz"],
            linestyle="--",
            linewidth=0.95,
            color=JT_COLORS[jt_ind],
            alpha=alpha,
            label=label,
        )



def overlay_p1_lines_by_jt(
    ax,
    p1_transitions,
    alpha=0.55,
    add_legend=True,
):
    """
    Show all four Jahn-Teller families with a consistent color code.

    JT-A = vermillion, JT-B = blue, JT-C = green, JT-D = purple.
    """
    for jt_ind, jt_label in enumerate(JT_LABELS):
        this_jt = [
            t for t in p1_transitions
            if t["jt_label"] == jt_label
        ]

        if len(this_jt) == 0:
            continue

        for line_ind, trans in enumerate(this_jt):
            ax.axvline(
                trans["freq_mhz"],
                linestyle="--",
                linewidth=1.05,
                color=JT_COLORS[jt_ind],
                alpha=alpha,
                label=(
                    jt_label
                    if add_legend and line_ind == 0
                    else None
                ),
            )


def save_key_figure_dm(
    fig,
    timestamp,
    repr_nv_name,
    suffix,
):
    """Save a key PNG through the standard data_manager workflow."""
    fig_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name + f"-deer-analysis-2p777GHz-{suffix}",
    )
    dm.save_figure(fig, fig_path)
    return fig_path.with_suffix(".png")




def prepare_counts(counts):
    counts = np.asarray(counts)

    if counts.ndim == 5:
        if counts.shape[0] != 1:
            raise ValueError(
                f"Expected num_exps=1, got counts shape {counts.shape}"
            )
        counts = counts[0]

    if counts.ndim == 3:
        counts = counts[..., np.newaxis]

    if counts.ndim != 4:
        raise ValueError(
            "Expected counts shape (NV, runs, steps, reps), "
            f"got {counts.shape}"
        )

    return counts


def get_nv_name(nv_list, nv_ind):
    try:
        return nv_list[nv_ind].name
    except Exception:
        return f"NV {nv_ind}"


def robust_limits(arr, percentile=98):
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return -1.0, 1.0

    vmax = np.nanpercentile(np.abs(finite), percentile)

    if (not np.isfinite(vmax)) or vmax <= 0:
        vmax = 1.0

    return -vmax, vmax


def main():
    kpl.init_kplotlib()

    print("\n============================================")
    print("Widefield DEER analysis")
    print("============================================")
    print(f"Loading:\n  {FILE_STEM}")

    data = dm.get_raw_data(
        file_stem=FILE_STEM,
        load_npz=True,
        use_cache=False,
    )

    # Timing metadata saved by the actual DEER acquisition.
    tau_ns = int(data.get("tau_ns", 18_000))
    nv_pi_ns = int(data.get("nv_pi_ns", 256))
    rf_pi_ns = float(data.get("rf_pi_ns", RF_PI_NS_FALLBACK))
    rf_pi_us = rf_pi_ns / 1000.0

    all_nv_list = data["nv_list"]
    all_counts = prepare_counts(data["counts"])

    # --------------------------------------------------------
    # Keep ONLY the 2.7773-GHz orientation family.
    #
    # TARGET_NV_INDICES are the original/current indices in the
    # 212-NV qnami list. Preserve those indices for all reporting.
    # --------------------------------------------------------
    selected_nv_indices = np.ascontiguousarray(
        TARGET_NV_INDICES,
        dtype=int,
    )

    if selected_nv_indices.size == 0:
        raise ValueError("TARGET_NV_INDICES is empty.")

    if np.min(selected_nv_indices) < 0 or np.max(selected_nv_indices) >= len(all_nv_list):
        raise IndexError(
            "TARGET_NV_INDICES contains an index outside the loaded NV list: "
            f"N_loaded={len(all_nv_list)}, "
            f"min={np.min(selected_nv_indices)}, "
            f"max={np.max(selected_nv_indices)}"
        )

    nv_list = [all_nv_list[int(i)] for i in selected_nv_indices]
    counts = np.ascontiguousarray(
        all_counts[selected_nv_indices, :, :, :]
    )

    num_nvs_counts, num_runs, num_steps_actual, num_reps = counts.shape
    num_nvs = len(nv_list)

    print("\nRaw dimensions")
    print(f"  all NVs loaded      : {len(all_nv_list)}")
    print(f"  selected orientation: {TARGET_GROUP} {TARGET_ORIENTATION}")
    print(f"  target ESR branch   : ~{TARGET_ESR_GHZ:.4f} GHz")
    print(f"  selected NVs        : {num_nvs}")
    print(f"  runs                : {num_runs}")
    print(f"  steps               : {num_steps_actual}")
    print(f"  reps                : {num_reps}")
    print(f"  original NV indices : {selected_nv_indices.tolist()}")

    if "freqs" not in data:
        raise KeyError("Dataset does not contain 'freqs'.")

    freqs_saved = np.asarray(data["freqs"], dtype=float)

    if num_steps_actual == 2 * len(freqs_saved):
        freqs_on = np.ascontiguousarray(freqs_saved)
    elif num_steps_actual == len(freqs_saved):
        print(
            "[INFO] Saved frequency array appears to contain "
            "the full interleaved sequence."
        )
        freqs_on = np.ascontiguousarray(freqs_saved[0::2])
    else:
        raise ValueError(
            "\nFrequency/count mismatch:\n"
            f"  len(data['freqs']) = {len(freqs_saved)}\n"
            f"  counts steps       = {num_steps_actual}\n"
            "Expected steps = 2 * number of physical DEER frequencies."
        )

    freqs_mhz = freqs_on * 1000.0
    num_freqs = len(freqs_on)

    print("\nFrequency axis")
    print(f"  points : {num_freqs}")
    print(
        f"  range  : {np.nanmin(freqs_mhz):.3f} "
        f"to {np.nanmax(freqs_mhz):.3f} MHz"
    )

    p1_transitions_full = get_significant_p1_transitions(
        P1_FULL_RANGE_MHZ[0],
        P1_FULL_RANGE_MHZ[1],
    )
    p1_transitions_in_range = get_significant_p1_transitions(
        float(np.nanmin(freqs_mhz)),
        float(np.nanmax(freqs_mhz)),
    )

    print("\nPredicted P1 transitions in measured range")
    for trans in p1_transitions_in_range:
        print(
            f"  {trans['jt_label']:<14s} "
            f"{trans['freq_mhz']:8.3f} MHz | "
            f"weight={trans['deer_weight']:.4f}"
        )

    print(
        f"\nFull simulated P1/JT range: "
        f"{P1_FULL_RANGE_MHZ[0]:.0f}-"
        f"{P1_FULL_RANGE_MHZ[1]:.0f} MHz"
    )

    on_inds = np.arange(0, num_steps_actual, 2)
    off_inds = np.arange(1, num_steps_actual, 2)

    if len(on_inds) != num_freqs:
        raise ValueError(
            f"Found {len(on_inds)} ON steps but {num_freqs} frequencies."
        )

    sig_counts = counts[:, :, on_inds, :]
    ref_counts = counts[:, :, off_inds, :]

    if DYNAMIC_THRESHOLD:
        print("\nApplying dynamic SCC thresholding...")
        sig_counts, ref_counts = widefield.threshold_counts(
            nv_list,
            sig_counts,
            ref_counts,
            dynamic_thresh=True,
        )

    avg_sig, avg_sig_ste, _ = widefield.average_counts(sig_counts)
    avg_ref, avg_ref_ste, _ = widefield.average_counts(ref_counts)

    avg_contrast, avg_contrast_ste = widefield.calc_contrast(
        sig_counts,
        ref_counts,
    )

    avg_snr, avg_snr_ste = widefield.calc_snr(
        sig_counts,
        ref_counts,
    )

    avg_sig = np.asarray(avg_sig, dtype=float)
    avg_sig_ste = np.asarray(avg_sig_ste, dtype=float)
    avg_ref = np.asarray(avg_ref, dtype=float)
    avg_ref_ste = np.asarray(avg_ref_ste, dtype=float)
    avg_contrast = np.asarray(avg_contrast, dtype=float)
    avg_contrast_ste = np.asarray(avg_contrast_ste, dtype=float)
    avg_snr = np.asarray(avg_snr, dtype=float)
    avg_snr_ste = np.asarray(avg_snr_ste, dtype=float)

    baseline = np.nanmedian(
        avg_contrast,
        axis=1,
        keepdims=True,
    )
    contrast_centered = avg_contrast - baseline

    peak_inds = np.zeros(num_nvs_counts, dtype=int)
    peak_freq_mhz = np.full(num_nvs_counts, np.nan)
    peak_contrast = np.full(num_nvs_counts, np.nan)
    peak_response = np.full(num_nvs_counts, np.nan)

    for nv_ind in range(num_nvs_counts):
        curve = contrast_centered[nv_ind]

        if not np.any(np.isfinite(curve)):
            continue

        peak_ind = int(np.nanargmax(np.abs(curve)))
        peak_inds[nv_ind] = peak_ind
        peak_freq_mhz[nv_ind] = freqs_mhz[peak_ind]
        peak_contrast[nv_ind] = curve[peak_ind]
        peak_response[nv_ind] = abs(curve[peak_ind])

    ranked_indices = np.ascontiguousarray(
        np.argsort(
            np.nan_to_num(
                peak_response,
                nan=-np.inf,
            )
        )[::-1]
    )

    # Map subset-local indices (0..101) back to the original 212-NV indices.
    ranked_original_indices = np.ascontiguousarray(
        selected_nv_indices[ranked_indices]
    )


    # --------------------------------------------------------
    # Save using the same data_manager workflow as experiments
    # --------------------------------------------------------
    timestamp = dm.get_time_stamp()

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    repr_nv_name = repr_nv_sig.name

    # IMPORTANT:
    # dm.get_file_path() returns the standard FULL data path
    # (typically ending in .txt), not just a directory.
    file_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name + "-deer-analysis-2p777GHz-groupA-fullP1",
    )

    # Put processed results into a normal raw-data dictionary.
    # dm.save_raw_data() creates the standard G:\nvdata directory
    # structure and handles compressed array storage.
    # Make every numeric array C-contiguous before handing it to
    # data_manager/orjson. Slices like arr[0::2] and reversed arrays
    # like arr[::-1] are otherwise often non-contiguous.
    freqs_on = np.ascontiguousarray(freqs_on)
    freqs_mhz = np.ascontiguousarray(freqs_mhz)
    avg_sig = np.ascontiguousarray(avg_sig)
    avg_sig_ste = np.ascontiguousarray(avg_sig_ste)
    avg_ref = np.ascontiguousarray(avg_ref)
    avg_ref_ste = np.ascontiguousarray(avg_ref_ste)
    avg_contrast = np.ascontiguousarray(avg_contrast)
    avg_contrast_ste = np.ascontiguousarray(avg_contrast_ste)
    contrast_centered = np.ascontiguousarray(contrast_centered)
    avg_snr = np.ascontiguousarray(avg_snr)
    avg_snr_ste = np.ascontiguousarray(avg_snr_ste)
    peak_freq_mhz = np.ascontiguousarray(peak_freq_mhz)
    peak_contrast = np.ascontiguousarray(peak_contrast)
    peak_response = np.ascontiguousarray(peak_response)
    ranked_indices = np.ascontiguousarray(ranked_indices)
    selected_nv_indices = np.ascontiguousarray(selected_nv_indices)
    ranked_original_indices = np.ascontiguousarray(ranked_original_indices)
    B_vec_G = np.ascontiguousarray(B_VEC)

    analysis_data = {
        "timestamp": timestamp,
        "source_file_stem": FILE_STEM,
        "nv_list": nv_list,
        "num_nvs": int(num_nvs_counts),
        "target_group": TARGET_GROUP,
        "target_orientation": TARGET_ORIENTATION,
        "target_esr_ghz": TARGET_ESR_GHZ,
        "selected_nv_indices": selected_nv_indices,
        "ranked_original_indices": ranked_original_indices,
        "B_vec_G": B_vec_G,
        "B_mag_G": float(np.linalg.norm(B_VEC)),
        "tau_ns": int(tau_ns),
        "nv_pi_ns": int(nv_pi_ns),
        "rf_pi_ns": float(rf_pi_ns),
        "p1_weight_threshold": float(P1_WEIGHT_THRESHOLD),
        "p1_full_range_mhz": np.ascontiguousarray(
            np.asarray(P1_FULL_RANGE_MHZ, dtype=float)
        ),
        "p1_transition_freq_mhz": np.ascontiguousarray(
            np.asarray(
                [t["freq_mhz"] for t in p1_transitions_full],
                dtype=float,
            )
        ),
        "p1_transition_weight": np.ascontiguousarray(
            np.asarray(
                [t["deer_weight"] for t in p1_transitions_full],
                dtype=float,
            )
        ),
        "p1_transition_jt_ind": np.ascontiguousarray(
            np.asarray(
                [t["jt_ind"] for t in p1_transitions_full],
                dtype=int,
            )
        ),
        "num_runs": int(num_runs),
        "num_reps": int(num_reps),
        "num_freqs": int(num_freqs),
        "freqs": freqs_on,
        "freq-units": "GHz",
        "freqs_mhz": freqs_mhz,
        "avg_sig": avg_sig,
        "avg_sig_ste": avg_sig_ste,
        "avg_ref": avg_ref,
        "avg_ref_ste": avg_ref_ste,
        "avg_contrast": avg_contrast,
        "avg_contrast_ste": avg_contrast_ste,
        "contrast_centered": contrast_centered,
        "avg_snr": avg_snr,
        "avg_snr_ste": avg_snr_ste,
        "peak_freq_mhz": peak_freq_mhz,
        "peak_contrast": peak_contrast,
        "peak_response": peak_response,
        "ranked_indices": ranked_indices,
    }

    # Compress all ndarray-heavy fields. This keeps the .txt metadata
    # small and moves the numerical arrays into the companion .npz,
    # matching the existing dm.save_raw_data workflow.
    keys_to_compress = [
        "freqs",
        "freqs_mhz",
        "avg_sig",
        "avg_sig_ste",
        "avg_ref",
        "avg_ref_ste",
        "avg_contrast",
        "avg_contrast_ste",
        "contrast_centered",
        "avg_snr",
        "avg_snr_ste",
        "peak_freq_mhz",
        "peak_contrast",
        "peak_response",
        "ranked_indices",
        "selected_nv_indices",
        "ranked_original_indices",
        "B_vec_G",
        "p1_full_range_mhz",
        "p1_transition_freq_mhz",
        "p1_transition_weight",
        "p1_transition_jt_ind",
    ]

    # This is the key old-style save call.
    dm.save_raw_data(
        analysis_data,
        file_path,
        keys_to_compress,
    )

    # At this point data_manager has created the target directory.
    # Use the exact same basename for PDF/CSV sidecars.
    file_path_str = str(file_path)
    base_no_ext, _ = os.path.splitext(file_path_str)

    pdf_path = base_no_ext + ".pdf"
    csv_path = base_no_ext + "-ranking.csv"

    print("\nOutput files")
    print(f"  DM data : {file_path}")
    print(f"  PDF     : {pdf_path}")
    print(f"  CSV     : {csv_path}")

    # --------------------------------------------------------
    # CSV ranking
    # --------------------------------------------------------

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(
        [
            "rank",
            "subset_index",
            "nv_index",
            "nv_name",
            "peak_frequency_MHz",
            "signed_peak_contrast",
            "absolute_response",
        ]
    )

    for rank, nv_ind in enumerate(ranked_indices, start=1):
        writer.writerow(
            [
                rank,
                int(nv_ind),
                int(selected_nv_indices[nv_ind]),
                get_nv_name(nv_list, nv_ind),
                float(peak_freq_mhz[nv_ind]),
                float(peak_contrast[nv_ind]),
                float(peak_response[nv_ind]),
            ]
        )

    dm.save_text(
        csv_buffer.getvalue(),
        csv_path,
        suffix=".csv",
    )

    num_source_files = (
        len(FILE_STEM) if isinstance(FILE_STEM, (list, tuple)) else 1
    )
    dataset_label = (
        f"{num_source_files} combined files | {num_runs} runs | "
        f"{freqs_mhz.min():.0f}-{freqs_mhz.max():.0f} MHz | "
        f"{TARGET_GROUP} ~{TARGET_ESR_GHZ:.4f} GHz"
    )

    with PdfPages(pdf_path) as pdf:
        # ====================================================
        # Page 0: experiment / P1 simulation summary
        # ====================================================

        B_mag = float(np.linalg.norm(B_VEC))
        B_hat = B_VEC / B_mag

        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(
            "Widefield DEER — Experiment / P1 Simulation Summary",
            fontsize=16,
            y=0.97,
        )

        summary_lines = [
            f"Source files: {num_source_files}",
            f"Combined runs: {num_runs}",
            (
                f"Selected NV family: {TARGET_GROUP} "
                f"{TARGET_ORIENTATION}"
            ),
            (
                f"Selected ODMR branch: "
                f"~{TARGET_ESR_GHZ:.4f} GHz"
            ),
            (
                f"Selected NVs: "
                f"{num_nvs_counts} / {len(all_nv_list)}"
            ),
            "",
            "Magnetic field:",
            (
                f"  B = [{B_VEC[0]:.2f}, "
                f"{B_VEC[1]:.2f}, "
                f"{B_VEC[2]:.2f}] G"
            ),
            f"  |B| = {B_mag:.3f} G",
            (
                f"  B_hat = [{B_hat[0]:.4f}, "
                f"{B_hat[1]:.4f}, "
                f"{B_hat[2]:.4f}]"
            ),
            "",
            (
                f"Measured P1 range: "
                f"{freqs_mhz.min():.1f}–"
                f"{freqs_mhz.max():.1f} MHz"
            ),
            f"Hahn tau = {tau_ns/1000:.3f} us "
            f"(total free evolution = {2*tau_ns/1000:.3f} us)",
            f"NV pi pulse = {nv_pi_ns:.0f} ns",
            f"RF/P1 pi pulse = {rf_pi_ns:.0f} ns",
            (
                "Rectangular-pulse bandwidth estimate: "
                f"{0.89/rf_pi_us:.2f} MHz"
            ),
            "",
            (
                f"Predicted P1/JT lines shown over full range: "
                f"{P1_FULL_RANGE_MHZ[0]:.0f}-"
                f"{P1_FULL_RANGE_MHZ[1]:.0f} MHz"
            ),
            f"Number of significant full-range lines: {len(p1_transitions_full)}",
        ]

        # fig, ax = plt.subplots(figsize=(11, 7.5))
        # for nv_ind in range(num_nvs_counts):
        #     ax.plot(
        #         freqs_mhz,
        #         avg_contrast[nv_ind],
        #         lw=0.6,
        #         alpha=0.20,
        #     )

        # median_curve = np.nanmedian(avg_contrast, axis=0)
        # ax.plot(freqs_mhz, median_curve, lw=2.0, label="Median")
        # overlay_p1_lines(
        #     ax,
        #     p1_transitions_in_range,
        #     label_once=True,
        #     alpha=0.50,
        # )
        # ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.5)
        # ax.set_xlabel("P1 / RF frequency (MHz)")
        # ax.set_ylabel("DEER contrast")
        # ax.set_title(
        #     f"Widefield DEER - selected 2.777-GHz orientation\n"
        #     f"{dataset_label} | N={num_nvs_counts}"
        # )
        # ax.legend()
        # ax.grid(alpha=0.2)
        # pdf.savefig(fig)

        # fig.text(
        #     0.07,
        #     0.91,
        #     "\n".join(summary_lines),
        #     va="top",
        #     ha="left",
        #     fontsize=10,
        #     family="monospace",
        # )
        # plt.show(block=True)
        # sys.exit()
        fig.text(
            0.07,
            0.28,
            (
                "Predicted line positions use the full 6x6 P1 "
                "Hamiltonian at the reconstructed field.\n"
                "Absolute DEER amplitudes are not fit here; "
                "the P1 weights are only relative."
            ),
            fontsize=9,
        )

        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 7.5))
        for nv_ind in range(num_nvs_counts):
            ax.plot(
                freqs_mhz,
                avg_contrast[nv_ind],
                lw=0.6,
                alpha=0.20,
            )

        median_curve = np.nanmedian(avg_contrast, axis=0)
        ax.plot(freqs_mhz, median_curve, lw=2.0, label="Median")
        overlay_p1_lines(
            ax,
            p1_transitions_in_range,
            label_once=True,
            alpha=0.50,
        )
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("P1 / RF frequency (MHz)")
        ax.set_ylabel("DEER contrast")
        ax.set_title(
            f"Widefield DEER - selected 2.777-GHz orientation\n"
            f"{dataset_label} | N={num_nvs_counts}"
        )
        ax.legend()
        ax.grid(alpha=0.2)
        pdf.savefig(fig)
        save_key_figure_dm(
            fig,
            timestamp,
            repr_nv_name,
            "all-selected-nvs",
        )
        plt.close(fig)

        p16 = np.nanpercentile(avg_contrast, 16, axis=0)
        p25 = np.nanpercentile(avg_contrast, 25, axis=0)
        p50 = np.nanpercentile(avg_contrast, 50, axis=0)
        p75 = np.nanpercentile(avg_contrast, 75, axis=0)
        p84 = np.nanpercentile(avg_contrast, 84, axis=0)

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.fill_between(
            freqs_mhz, p16, p84, alpha=0.15, label="16-84 percentile"
        )
        ax.fill_between(
            freqs_mhz, p25, p75, alpha=0.30, label="IQR"
        )
        ax.plot(
            freqs_mhz, p50, marker="o", markersize=2.5, lw=1.5, label="Median"
        )
        overlay_p1_lines(
            ax,
            p1_transitions_in_range,
            label_once=True,
            alpha=0.50,
        )
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("P1 / RF frequency (MHz)")
        ax.set_ylabel("DEER contrast")
        ax.set_title(
            f"Median DEER spectrum - {TARGET_GROUP} "
            f"(~{TARGET_ESR_GHZ:.4f} GHz, N={num_nvs_counts})"
        )
        ax.legend()
        ax.grid(alpha=0.2)
        pdf.savefig(fig)
        save_key_figure_dm(
            fig,
            timestamp,
            repr_nv_name,
            "median-spectrum",
        )
        plt.close(fig)

        # ====================================================
        # Full-range experiment + P1/JT simulation overview
        # ====================================================

        fig, (ax_exp, ax_stick) = plt.subplots(
            2,
            1,
            figsize=(11, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.5]},
        )

        # Experimental data exist only over the measured band, but the
        # common x-axis is deliberately extended to 10-300 MHz so the
        # complete P1/JT structure is visible.
        ax_exp.plot(
            freqs_mhz,
            p50,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label="Experimental median",
        )
        ax_exp.fill_between(
            freqs_mhz,
            p25,
            p75,
            alpha=0.20,
            label="Experimental IQR",
        )
        overlay_p1_lines_by_jt(
            ax_exp,
            p1_transitions_full,
            alpha=0.32,
            add_legend=True,
        )
        ax_exp.set_ylabel("DEER contrast")
        ax_exp.set_title(
            "Measured DEER band with full 10-300 MHz P1/JT prediction"
        )
        ax_exp.legend(fontsize=8.5, ncol=2, frameon=True)
        ax_exp.grid(alpha=0.15)

        for jt_ind, jt_label in enumerate(JT_LABELS):
            this_jt = [
                t for t in p1_transitions_full
                if t["jt_label"] == jt_label
            ]
            for trans in this_jt:
                ax_stick.vlines(
                    trans["freq_mhz"],
                    0,
                    trans["deer_weight"],
                    color=JT_COLORS[jt_ind],
                    linewidth=1.8,
                    alpha=0.90,
                )

            if len(this_jt) > 0:
                ax_stick.plot(
                    [],
                    [],
                    color=JT_COLORS[jt_ind],
                    linewidth=2.0,
                    label=jt_label,
                )

        ax_stick.set_xlim(*P1_FULL_RANGE_MHZ)
        ax_stick.set_xlabel("P1 / RF frequency (MHz)")
        ax_stick.set_ylabel("Relative\nP1 weight")
        ax_stick.legend(fontsize=8.5, ncol=2, frameon=True)
        ax_stick.grid(alpha=0.12)

        fig.tight_layout()
        pdf.savefig(fig)
        save_key_figure_dm(
            fig,
            timestamp,
            repr_nv_name,
            "experiment-vs-full-P1-JT",
        )
        plt.close(fig)

        heatmap_data = contrast_centered[ranked_indices]
        vmin, vmax = robust_limits(heatmap_data, percentile=98)

        fig, ax = plt.subplots(figsize=(11, 8))
        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            origin="lower",
            extent=[
                freqs_mhz[0],
                freqs_mhz[-1],
                0,
                num_nvs_counts,
            ],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Baseline-subtracted DEER contrast")
        ax.set_xlabel("P1 / RF frequency (MHz)")
        ax.set_ylabel("NVs sorted by DEER response")
        ax.set_title("DEER contrast heatmap")
        pdf.savefig(fig)
        save_key_figure_dm(
            fig,
            timestamp,
            repr_nv_name,
            "ranked-heatmap",
        )
        plt.close(fig)

        top_n = min(TOP_N, num_nvs_counts)
        top_indices = ranked_indices[:top_n]

        fig, ax = plt.subplots(figsize=(11, 7.5))
        for nv_ind in top_indices:
            ax.plot(
                freqs_mhz,
                contrast_centered[nv_ind],
                lw=1.0,
                alpha=0.8,
                label=f"NV {int(selected_nv_indices[nv_ind])}",
            )

        overlay_p1_lines(
            ax,
            p1_transitions_in_range,
            label_once=True,
            alpha=0.45,
        )
        ax.axhline(0, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("P1 / RF frequency (MHz)")
        ax.set_ylabel("Baseline-subtracted DEER contrast")
        ax.set_title(
            f"Top {top_n} NVs ranked by maximum |DEER response|"
        )
        if top_n <= 20:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.2)
        pdf.savefig(fig)
        save_key_figure_dm(
            fig,
            timestamp,
            repr_nv_name,
            "top-nvs",
        )
        plt.close(fig)

        ncols = 3
        nrows = int(np.ceil(NV_PER_PAGE / ncols))

        for start in range(0, num_nvs_counts, NV_PER_PAGE):
            stop = min(start + NV_PER_PAGE, num_nvs_counts)
            inds = list(range(start, stop))

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(11, 3.0 * nrows),
                sharex=True,
            )

            axes = np.asarray(axes).reshape(-1)

            for ax in axes:
                ax.set_visible(False)

            for ax, nv_ind in zip(axes, inds):
                ax.set_visible(True)

                y = avg_contrast[nv_ind]
                yerr = avg_contrast_ste[nv_ind]

                ax.errorbar(
                    freqs_mhz,
                    y,
                    yerr=np.abs(yerr),
                    marker="o",
                    markersize=2,
                    linewidth=0.8,
                    capsize=1,
                )
                overlay_p1_lines(
                    ax,
                    p1_transitions_in_range,
                    label_once=False,
                    alpha=0.20,
                )

                peak_ind = peak_inds[nv_ind]

                ax.axvline(
                    freqs_mhz[peak_ind],
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.6,
                )

                ax.axhline(
                    np.nanmedian(y),
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.5,
                )

                ax.set_title(
                    f"NV {int(selected_nv_indices[nv_ind])}\n"
                    f"peak={peak_freq_mhz[nv_ind]:.1f} MHz, "
                                        f"|C|={peak_response[nv_ind]:.3g}",
                    fontsize=9,
                )

                ax.grid(alpha=0.15)

            fig.supxlabel("P1 / RF frequency (MHz)")
            fig.supylabel("DEER contrast")

            fig.suptitle(
                f"{dataset_label}\n"
                f"Individual selected-NV spectra "
                f"{start}-{stop - 1} (subset positions)",
                fontsize=12,
            )


            pdf.savefig(fig)
            plt.close(fig)

    print("\n============================================")
    print("Strongest DEER responses")
    print("============================================")

    for rank, nv_ind in enumerate(
        ranked_indices[:min(20, num_nvs_counts)],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"NV {int(selected_nv_indices[nv_ind]):3d} | "
            f"{peak_freq_mhz[nv_ind]:7.2f} MHz | "
            f"C={peak_contrast[nv_ind]:+.5f} | "
            f"|C|={peak_response[nv_ind]:.5f}"
        )

    print("\nSaved successfully:")
    print(f"  DM data : {file_path}")
    print(f"  PDF     : {pdf_path}")
    print(f"  CSV     : {csv_path}")

    plt.close("all")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nDEER analysis failed:\n")
        print(traceback.format_exc())
        raise
