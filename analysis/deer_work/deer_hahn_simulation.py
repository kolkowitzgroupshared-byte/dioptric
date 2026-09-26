#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
P1-center DEER spectrum simulator for a ~52.4 G magnetic field.

Creates ONE combined multi-page PDF with:
    1. Magnetic-field / NV-orientation summary
    2. P1 stick spectrum + broadened DEER spectrum
    3. Significant-transition table
    4. Concentration comparison
    5. P1 energy levels for all four Jahn-Teller axes

Physics:
    - Full 6x6 P1 Hamiltonian (S=1/2 x I=1)
    - All four P1 Jahn-Teller orientations
    - Full spin-1 NV Hamiltonian
    - Field-axis electron-spin-flip weight for P1 transitions
    - Gaussian phenomenological broadening

Field configuration:
    B = (-48.55, -18.75, -5.97) G
    |B| ~= 52.39 G
    selected crystallographic NV axis = (1, 1, -1)
    equivalent simulator axis = (-1, -1, 1)

Notes:
    - P1 line POSITIONS are the main quantitative output.
    - Relative / absolute DEER amplitudes remain phenomenological because
      the NV-P1 dipolar geometry is not explicitly modeled.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.linalg import eigh

from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# USER CONFIGURATION
# =============================================================================

B_VEC = np.array([-48.55, -18.75, -5.97], dtype=float)  # Gauss

# Calibrated here so the selected NV family is ~2.7773 GHz for this B vector.
D_NV = 2870.4  # MHz

# P1 bath pulse duration.
# P1/bath pi pulse used for the spectral-width model.
# For a true pi pulse this corresponds to f_Rabi = 5 MHz.
MW_PULSE_US = 0.100

FREQ_RANGE_MHZ = (10.0, 300.0)
N_POINTS = 5000

CONCENTRATIONS_PPB = [5, 25, 75, 100, 200, 500, 1000]
REFERENCE_CONC_PPB = 5

# Show only transitions above this DEER-weight threshold in summary/table.
SIGNIFICANT_WEIGHT = 0.005

# Keep the same width on every PDF page; page heights may vary.
REPORT_WIDTH = 8.5
SUMMARY_FIGSIZE = (REPORT_WIDTH, 13.5)
SPECTRUM_FIGSIZE = (REPORT_WIDTH, 8.0)
TABLE_FIGSIZE = (REPORT_WIDTH, 11.0)
CONCENTRATION_FIGSIZE = (REPORT_WIDTH, 7.0)
ENERGY_FIGSIZE = (REPORT_WIDTH, 8.5)

# Selected NV orientation used for the NV-frequency calculation.
SELECTED_NV_AXIS = np.array([-1, -1, 1], dtype=float) / np.sqrt(3)
SELECTED_NV_LABEL = "selected NV axis: (1,1,-1) ≡ (-1,-1,1)"
TARGET_NV_GHZ = 2.7773


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

GAMMA_E = 2.8025       # electron gyromagnetic ratio [MHz/G]
GAMMA_N14 = 3.077e-4   # 14N nuclear gyromagnetic ratio [MHz/G]

A_PAR = 114.0           # MHz
A_PERP = 81.3           # MHz
P_PAR = -3.97           # MHz

CARBON_DENSITY = 1.764e23  # cm^-3


# =============================================================================
# CRYSTALLOGRAPHIC AXES
# =============================================================================

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
    "A [111]",
    "B [-1-11]",
    "C [-11-1]",
    "D [1-1-1]",
]

# Consistent colorblind-friendly colors for the four P1/JT families.
JT_COLORS = [
    "#D55E00",  # JT-A: vermillion
    "#0072B2",  # JT-B: blue
    "#009E73",  # JT-C: green
    "#CC79A7",  # JT-D: purple
]

NV_AXES = JT_AXES.copy()

NV_LABELS = [
    "NV [111]",
    "NV [-1-11]",
    "NV [-11-1]",
    "NV [1-1-1]",
]


# =============================================================================
# SPIN OPERATORS
# =============================================================================

def build_spin_operators():
    """P1 operators for S=1/2 x I=1."""
    sx_half = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy_half = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz_half = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    I2 = np.eye(2, dtype=complex)

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
    I3 = np.eye(3, dtype=complex)

    Sx = np.kron(sx_half, I3)
    Sy = np.kron(sy_half, I3)
    Sz = np.kron(sz_half, I3)

    Ix = np.kron(I2, ix_one)
    Iy = np.kron(I2, iy_one)
    Iz = np.kron(I2, iz_one)

    return [Sx, Sy, Sz], [Ix, Iy, Iz]


def build_nv_spin1_operators():
    """NV ground-state spin-1 operators."""
    sq2 = np.sqrt(2)

    Sx = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        dtype=complex,
    ) / sq2

    Sy = np.array(
        [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]],
        dtype=complex,
    ) / sq2

    Sz = np.diag([1.0, 0.0, -1.0]).astype(complex)

    return Sx, Sy, Sz


# =============================================================================
# P1 HAMILTONIAN
# =============================================================================

def build_p1_hamiltonian(B_vec, jt_axis, S_ops, I_ops):
    """
    H = gamma_e B.S - gamma_N B.I
        + A_perp S.I + (A_par-A_perp)(n.S)(n.I)
        + P[(n.I)^2 - I(I+1)/3]
    """
    Sx, Sy, Sz = S_ops
    Ix, Iy, Iz = I_ops

    n = jt_axis
    dim = 6

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
        - (2.0 / 3.0) * np.eye(dim, dtype=complex)
    )

    return H_ez + H_nz + H_hf + H_q


def diagonalize_p1(B_vec, jt_axis, S_ops, I_ops):
    H = build_p1_hamiltonian(B_vec, jt_axis, S_ops, I_ops)
    return eigh(H)


def compute_p1_transitions(
    B_vec,
    jt_axis,
    S_ops,
    I_ops,
    intensity_threshold=1e-4,
):
    """
    Compute P1 transitions.

    Microwave intensity:
        sum of matrix elements for two orthogonal directions transverse to B.

    DEER spin-flip factor:
        change in <S_B>, with S_B = B_hat . S.

    This replaces the previous crystal-z-only Delta<Sz> measure.
    """
    Sx, Sy, Sz = S_ops

    evals, evecs = diagonalize_p1(
        B_vec,
        jt_axis,
        S_ops,
        I_ops,
    )

    B_hat = B_vec / np.linalg.norm(B_vec)

    # Two orthonormal MW polarization directions perpendicular to B
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

    # Important correction:
    # electron-spin change is evaluated along the static-field axis.
    S_B = (
        B_hat[0] * Sx
        + B_hat[1] * Sy
        + B_hat[2] * Sz
    )

    transitions = []

    for i in range(6):
        for j in range(i + 1, 6):
            freq = float(evals[j] - evals[i])

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

            if intensity > intensity_threshold:
                transitions.append(
                    {
                        "freq": freq,
                        "intensity": intensity,
                        "delta_SB": delta_SB,
                        "deer_weight": intensity * delta_SB,
                        "states": (i, j),
                    }
                )

    return transitions


# =============================================================================
# NV HAMILTONIAN
# =============================================================================

def compute_nv_transitions(B_vec, nv_axis):
    """Full spin-1 NV Hamiltonian in the local NV frame."""
    Sx_nv, Sy_nv, Sz_nv = build_nv_spin1_operators()

    B_par = float(np.dot(B_vec, nv_axis))
    B_perp_vec = B_vec - B_par * nv_axis
    B_perp = float(np.linalg.norm(B_perp_vec))

    z_nv = nv_axis

    if B_perp > 1e-12:
        x_nv = B_perp_vec / B_perp
    else:
        if abs(z_nv[2]) < 0.9:
            x_nv = np.cross(z_nv, [0, 0, 1])
        else:
            x_nv = np.cross(z_nv, [1, 0, 0])
        x_nv /= np.linalg.norm(x_nv)

    y_nv = np.cross(z_nv, x_nv)

    Bx_nv = float(np.dot(B_vec, x_nv))
    By_nv = float(np.dot(B_vec, y_nv))
    Bz_nv = float(np.dot(B_vec, z_nv))

    H_nv = D_NV * (Sz_nv @ Sz_nv) + GAMMA_E * (
        Bx_nv * Sx_nv
        + By_nv * Sy_nv
        + Bz_nv * Sz_nv
    )

    evals, evecs = eigh(H_nv)

    overlaps_0 = np.abs(evecs[1, :]) ** 2
    idx_0 = int(np.argmax(overlaps_0))

    other_idx = [k for k in range(3) if k != idx_0]

    f_transitions = sorted(
        [
            abs(float(evals[k] - evals[idx_0]))
            for k in other_idx
        ]
    )

    return {
        "B_par": B_par,
        "B_perp": B_perp,
        "f_lower": f_transitions[0],
        "f_upper": f_transitions[1],
        "evals": evals,
    }


# =============================================================================
# LINE SHAPE / DEER SPECTRUM
# =============================================================================

def gaussian_lineshape(f, f0, sigma):
    return np.exp(
        -0.5 * ((f - f0) / sigma) ** 2
    ) / (sigma * np.sqrt(2 * np.pi))


def fwhm_to_sigma(fwhm):
    return fwhm / (
        2 * np.sqrt(2 * np.log(2))
    )


def concentration_to_linewidth(
    conc_ppb,
    mw_pulse_us=MW_PULSE_US,
):
    """
    Phenomenological total FWHM [MHz].

    Dipolar:
        4.5 MHz at 1 ppm, linear in concentration.

    Rectangular-pulse bandwidth:
        approx 0.89/t_pulse.

    Intrinsic T2* contribution:
        ~0.0064 MHz.
    """
    gamma_dipolar = 4.5e-3 * conc_ppb
    gamma_mw = 0.89 / mw_pulse_us
    gamma_t2star = 0.0064

    return np.sqrt(
        gamma_dipolar**2
        + gamma_mw**2
        + gamma_t2star**2
    )


def build_deer_spectrum(
    B_vec,
    freq_range,
    concentrations_ppb,
    mw_pulse_us=MW_PULSE_US,
    n_points=N_POINTS,
):
    S_ops, I_ops = build_spin_operators()

    f_min, f_max = freq_range
    freqs = np.linspace(f_min, f_max, n_points)

    all_transitions = {}

    for jt_idx, (jt_axis, jt_label) in enumerate(
        zip(JT_AXES, JT_LABELS)
    ):
        trans = compute_p1_transitions(
            B_vec,
            jt_axis,
            S_ops,
            I_ops,
            intensity_threshold=1e-4,
        )

        for t in trans:
            t["jt_label"] = jt_label
            t["jt_idx"] = jt_idx

        all_transitions[jt_label] = trans

    spectra = {}

    for conc in concentrations_ppb:
        fwhm = concentration_to_linewidth(
            conc,
            mw_pulse_us,
        )
        sigma = fwhm_to_sigma(fwhm)

        spectrum = np.ones_like(freqs)

        for jt_label, trans_list in all_transitions.items():
            jt_weight = 0.25

            for t in trans_list:
                dip_amplitude = (
                    t["deer_weight"]
                    * jt_weight
                )

                # Phenomenological concentration scaling.
                conc_scale = conc / 75.0
                scaled_depth = dip_amplitude * conc_scale
                scaled_depth = min(scaled_depth, 0.5)

                peak = gaussian_lineshape(
                    freqs,
                    t["freq"],
                    sigma,
                )

                peak_max = (
                    1.0
                    / (
                        sigma
                        * np.sqrt(2 * np.pi)
                    )
                )

                if peak_max > 0:
                    spectrum -= (
                        scaled_depth
                        * peak
                        / peak_max
                    )

        spectra[conc] = np.clip(
            spectrum,
            0,
            1,
        )

    return freqs, all_transitions, spectra


# =============================================================================
# STANDARD DATA-MANAGER SAVE HELPERS
# =============================================================================

def save_sim_figure_dm(fig, timestamp, suffix):
    """Save a key simulation figure through utils.data_manager."""
    fig_path = dm.get_file_path(
        __file__,
        timestamp,
        f"p1-deer-52G-{suffix}",
    )
    dm.save_figure(fig, fig_path)
    return fig_path.with_suffix(".png")


# =============================================================================
# REPORT HELPERS
# =============================================================================

def get_significant_transitions(all_transitions):
    rows = []

    for jt_label in JT_LABELS:
        for t in all_transitions[jt_label]:
            if t["deer_weight"] >= SIGNIFICANT_WEIGHT:
                rows.append(
                    {
                        "JT": jt_label,
                        "freq": t["freq"],
                        "intensity": t["intensity"],
                        "delta_SB": t["delta_SB"],
                        "weight": t["deer_weight"],
                        "states": t["states"],
                    }
                )

    rows.sort(key=lambda row: row["freq"])
    return rows


def add_summary_page(pdf, all_transitions, timestamp):
    B_mag = float(np.linalg.norm(B_VEC))
    B_hat = B_VEC / B_mag

    selected = compute_nv_transitions(
        B_VEC,
        SELECTED_NV_AXIS,
    )

    all_nv_info = [
        (
            label,
            compute_nv_transitions(B_VEC, axis),
        )
        for axis, label in zip(
            NV_AXES,
            NV_LABELS,
        )
    ]

    rows = get_significant_transitions(
        all_transitions
    )

    strongest = sorted(
        rows,
        key=lambda r: r["weight"],
        reverse=True,
    )[:12]

    # A taller first page with explicitly separated vertical sections.
    fig = plt.figure(figsize=(REPORT_WIDTH, 14.5))

    fig.suptitle(
        "P1 DEER Simulation — 52 G Field / Full P1 Hamiltonian",
        fontsize=16,
        y=0.985,
    )

    # ======================================================
    # Section 1: model equations
    # ======================================================
    ax_model = fig.add_axes([0.07, 0.735, 0.86, 0.215])
    ax_model.axis("off")

    ax_model.text(
        0.0,
        1.00,
        "P1 spin Hamiltonian",
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    ax_model.text(
        0.0,
        0.82,
        (
            r"$H_{\mathrm{P1}}="
            r"\gamma_e\,\mathbf{B}\!\cdot\!\mathbf{S}"
            r"-\gamma_{^{14}\mathrm{N}}\,\mathbf{B}\!\cdot\!\mathbf{I}"
            r"+A_{\perp}\,\mathbf{S}\!\cdot\!\mathbf{I}$"
        ),
        fontsize=12,
        va="top",
    )

    ax_model.text(
        0.0,
        0.66,
        (
            r"$\qquad"
            r"+(A_{\parallel}-A_{\perp})"
            r"(\mathbf{n}\!\cdot\!\mathbf{S})"
            r"(\mathbf{n}\!\cdot\!\mathbf{I})"
            r"+P_{\parallel}"
            r"\left[(\mathbf{n}\!\cdot\!\mathbf{I})^2-\frac{2}{3}\right]$"
        ),
        fontsize=12,
        va="top",
    )

    ax_model.text(
        0.0,
        0.43,
        "Relative DEER transition weight",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    ax_model.text(
        0.0,
        0.29,
        (
            r"$w_{ij}\propto "
            r"\left("
            r"|\langle j|S_{\perp 1}|i\rangle|^2"
            r"+|\langle j|S_{\perp 2}|i\rangle|^2"
            r"\right)"
            r"\left|"
            r"\langle j|S_B|j\rangle"
            r"-\langle i|S_B|i\rangle"
            r"\right|$"
        ),
        fontsize=11.3,
        va="top",
    )

    ax_model.text(
        0.0,
        0.06,
        (
            r"$\Gamma_{\mathrm{FWHM}}="
            r"\sqrt{"
            r"\left(4.5\times10^{-3}[{\rm P1}]\right)^2"
            r"+\left(\frac{0.89}{t_{\pi}}\right)^2"
            r"+\left(0.0064\right)^2"
            r"}\ \mathrm{MHz}$"
        ),
        fontsize=11.2,
        va="bottom",
    )

    # ======================================================
    # Section 2: symbols / physical meaning
    # ======================================================
    ax_symbols = fig.add_axes([0.07, 0.565, 0.86, 0.145])
    ax_symbols.axis("off")

    ax_symbols.text(
        0.0,
        1.00,
        "Symbols and physical terms",
        fontsize=11,
        fontweight="bold",
        va="top",
    )

    left_terms = [
        r"$\mathbf{S}$: P1 electron spin, $S=1/2$",
        r"$\mathbf{I}$: $^{14}$N nuclear spin, $I=1$",
        r"$\mathbf{B}$: applied magnetic field",
        r"$\mathbf{n}$: P1 Jahn-Teller symmetry axis",
        r"$\gamma_e$: electron gyromagnetic ratio",
        r"$\gamma_{^{14}\mathrm{N}}$: nuclear gyromagnetic ratio",
    ]

    right_terms = [
        r"$A_{\parallel},A_{\perp}$: axial/transverse hyperfine coupling",
        r"$P_{\parallel}$: $^{14}$N quadrupole interaction",
        r"$S_B=\hat{\mathbf{B}}\!\cdot\!\mathbf{S}$: spin along field",
        r"$S_{\perp1},S_{\perp2}$: MW-driving spin components",
        r"$[{\rm P1}]$: P1 concentration in ppb",
        r"$t_{\pi}$: bath-spin $\pi$-pulse duration",
    ]

    ax_symbols.text(
        0.00,
        0.78,
        "\n".join(left_terms),
        fontsize=8.6,
        va="top",
        linespacing=1.35,
    )

    ax_symbols.text(
        0.51,
        0.78,
        "\n".join(right_terms),
        fontsize=8.6,
        va="top",
        linespacing=1.35,
    )

    ax_symbols.text(
        0.0,
        0.02,
        (
            r"$A_{\parallel}\neq A_{\perp}$ produces anisotropic hyperfine "
            r"splitting. The four allowed $\mathbf{n}$ directions give the "
            r"four JT families. The relative DEER weight favors transitions "
            r"that are both MW-addressable and change the electron-spin "
            r"projection along the static field."
        ),
        fontsize=8.2,
        va="bottom",
        wrap=True,
    )

    # ======================================================
    # Section 3: field + P1/NV numerical parameters
    # ======================================================
    ax_params = fig.add_axes([0.07, 0.335, 0.86, 0.200])
    ax_params.axis("off")

    param_lines = [
        "MAGNETIC FIELD",
        f"  B = [{B_VEC[0]:.2f}, {B_VEC[1]:.2f}, {B_VEC[2]:.2f}] G",
        f"  |B| = {B_mag:.3f} G",
        (
            "  B_hat = "
            f"[{B_hat[0]:.4f}, {B_hat[1]:.4f}, {B_hat[2]:.4f}]"
        ),
        f"  gamma_e|B| = {GAMMA_E*B_mag:.2f} MHz",
        "",
        "SELECTED NV ORIENTATION",
        f"  {SELECTED_NV_LABEL}",
        f"  D_NV = {D_NV:.2f} MHz",
        f"  B_parallel = {selected['B_par']:.2f} G",
        f"  B_perp = {selected['B_perp']:.2f} G",
        f"  lower ODMR = {selected['f_lower']/1000:.6f} GHz",
        f"  upper ODMR = {selected['f_upper']/1000:.6f} GHz",
        "",
        "P1 PARAMETERS",
        f"  A_parallel = {A_PAR:.1f} MHz",
        f"  A_perp = {A_PERP:.1f} MHz",
        f"  P_parallel = {P_PAR:.2f} MHz",
        f"  bath pi-pulse duration = {MW_PULSE_US*1000:.0f} ns",
        (
            "  rectangular-pulse bandwidth estimate = "
            f"{0.89/MW_PULSE_US:.2f} MHz"
        ),
    ]

    ax_params.text(
        0.0,
        1.0,
        "\n".join(param_lines),
        va="top",
        ha="left",
        fontsize=8.2,
        family="monospace",
        linespacing=1.22,
    )

    # ======================================================
    # Section 4: NV orientation table
    # ======================================================
    table_data = []

    for label, info in all_nv_info:
        table_data.append(
            [
                label,
                f"{info['B_par']:.2f}",
                f"{info['f_lower']/1000:.4f}",
                f"{info['f_upper']/1000:.4f}",
            ]
        )

    ax_table = fig.add_axes([0.07, 0.180, 0.86, 0.120])
    ax_table.axis("off")

    ax_table.text(
        0.0,
        1.08,
        "NV orientation frequencies",
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
    )

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=[
            "NV axis",
            "B_parallel [G]",
            "lower [GHz]",
            "upper [GHz]",
        ],
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1, 1.22)

    # ======================================================
    # Section 5: strongest P1 transitions
    # ======================================================
    ax_trans = fig.add_axes([0.07, 0.035, 0.86, 0.105])
    ax_trans.axis("off")

    strong_text = ["Strongest P1 transitions:"]
    for row in strongest:
        strong_text.append(
            f"{row['JT']:<11s} "
            f"{row['freq']:7.2f} MHz   "
            f"weight={row['weight']:.4f}"
        )

    ax_trans.text(
        0.0,
        1.0,
        "\n".join(strong_text),
        va="top",
        fontsize=7.2,
        family="monospace",
        linespacing=1.15,
    )

    pdf.savefig(fig)
    save_sim_figure_dm(
        fig,
        timestamp,
        "summary",
    )
    plt.close(fig)



def add_spectrum_page(
    pdf,
    freqs,
    all_transitions,
    spectra,
    timestamp,
):
    B_mag = np.linalg.norm(B_VEC)
    larmor = GAMMA_E * B_mag

    ref_conc = (
        REFERENCE_CONC_PPB
        if REFERENCE_CONC_PPB in spectra
        else min(CONCENTRATIONS_PPB)
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=SPECTRUM_FIGSIZE,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 2],
        },
    )

    for jt_ind, jt_label in enumerate(JT_LABELS):
        this_jt = [
            t for t in all_transitions[jt_label]
            if t["deer_weight"] >= SIGNIFICANT_WEIGHT
        ]

        for t in this_jt:
            ax1.vlines(
                t["freq"],
                0,
                t["deer_weight"],
                color=JT_COLORS[jt_ind],
                linewidth=1.8,
                alpha=0.90,
            )

        if len(this_jt) > 0:
            ax1.plot(
                [],
                [],
                color=JT_COLORS[jt_ind],
                linewidth=2.0,
                label=f"JT-{jt_label}",
            )

    ax1.axvline(
        larmor,
        ls="--",
        lw=0.8,
        alpha=0.5,
        label="bare electron Larmor",
    )
    ax1.set_ylabel("DEER weight")
    ax1.set_title(
        "Full 10-300 MHz P1 transition spectrum — all four JT families"
    )
    ax1.legend(fontsize=8.5, ncol=2, frameon=True)
    ax1.grid(alpha=0.15)

    ax2.plot(
        freqs,
        spectra[ref_conc],
        lw=1.4,
    )
    ax2.fill_between(
        freqs,
        spectra[ref_conc],
        1,
        alpha=0.12,
    )
    ax2.axvline(
        larmor,
        ls="--",
        lw=0.8,
        alpha=0.5,
    )

    ax2.set_xlabel("P1 drive frequency (MHz)")
    ax2.set_ylabel("Simulated DEER signal")
    ax2.set_title(
        f"Broadened spectrum | "
        f"[P1]={ref_conc} ppb | "
        f"t_pi={MW_PULSE_US*1000:.0f} ns"
    )
    ax2.grid(alpha=0.15)

    fig.tight_layout()
    pdf.savefig(fig)
    save_sim_figure_dm(
        fig,
        timestamp,
        "full-JT-spectrum",
    )
    plt.close(fig)


def add_transition_table_pages(
    pdf,
    all_transitions,
):
    rows = get_significant_transitions(
        all_transitions
    )

    rows_per_page = 28

    for start in range(
        0,
        len(rows),
        rows_per_page,
    ):
        chunk = rows[
            start : start + rows_per_page
        ]

        fig, ax = plt.subplots(
            figsize=TABLE_FIGSIZE
        )
        ax.axis("off")

        cell_text = []

        for row in chunk:
            cell_text.append(
                [
                    row["JT"],
                    f"{row['freq']:.3f}",
                    f"{row['intensity']:.4f}",
                    f"{row['delta_SB']:.4f}",
                    f"{row['weight']:.4f}",
                    f"{row['states'][0]}-{row['states'][1]}",
                ]
            )

        table = ax.table(
            cellText=cell_text,
            colLabels=[
                "JT axis",
                "Freq [MHz]",
                "MW intensity",
                "Δ<S_B>",
                "DEER weight",
                "states",
            ],
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.32)

        ax.set_title(
            "Significant P1 transitions "
            f"({start+1}–{start+len(chunk)} of {len(rows)})",
            fontsize=14,
            pad=20,
        )

        pdf.savefig(fig)
        plt.close(fig)


def add_concentration_page(
    pdf,
    freqs,
    spectra,
    timestamp,
):
    fig, ax = plt.subplots(
        figsize=CONCENTRATION_FIGSIZE
    )

    for conc in CONCENTRATIONS_PPB:
        ax.plot(
            freqs,
            spectra[conc],
            lw=1.0,
            label=(
                f"{conc} ppb "
                f"(FWHM "
                f"{concentration_to_linewidth(conc):.2f} MHz)"
            ),
        )

    ax.set_xlabel("P1 drive frequency (MHz)")
    ax.set_ylabel("Simulated DEER signal")
    ax.set_title(
        "Phenomenological concentration / linewidth comparison"
    )
    ax.legend(
        fontsize=8,
        ncol=2,
    )
    ax.grid(alpha=0.15)

    fig.tight_layout()
    pdf.savefig(fig)
    save_sim_figure_dm(
        fig,
        timestamp,
        "concentration-comparison",
    )
    plt.close(fig)


def add_energy_levels_page(
    pdf,
    timestamp,
):
    S_ops, I_ops = build_spin_operators()
    B_hat = B_VEC / np.linalg.norm(B_VEC)

    S_B = (
        B_hat[0] * S_ops[0]
        + B_hat[1] * S_ops[1]
        + B_hat[2] * S_ops[2]
    )

    I_B = (
        B_hat[0] * I_ops[0]
        + B_hat[1] * I_ops[1]
        + B_hat[2] * I_ops[2]
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=ENERGY_FIGSIZE,
    )

    axes = axes.reshape(-1)

    for ax, jt_axis, jt_label in zip(
        axes,
        JT_AXES,
        JT_LABELS,
    ):
        evals, evecs = diagonalize_p1(
            B_VEC,
            jt_axis,
            S_ops,
            I_ops,
        )

        SB_exp = [
            float(
                np.real(
                    evecs[:, k].conj()
                    @ S_B
                    @ evecs[:, k]
                )
            )
            for k in range(6)
        ]

        IB_exp = [
            float(
                np.real(
                    evecs[:, k].conj()
                    @ I_B
                    @ evecs[:, k]
                )
            )
            for k in range(6)
        ]

        energy0 = float(np.min(evals))
        rel_e = evals - energy0

        for k in range(6):
            ax.hlines(
                k,
                0,
                rel_e[k],
                linewidth=5,
                color=JT_COLORS[JT_LABELS.index(jt_label)],
                alpha=0.78,
            )

            ax.text(
                rel_e[k] + 2,
                k,
                (
                    f"{rel_e[k]:.1f} MHz\n"
                    f"<S_B>={SB_exp[k]:+.2f}, "
                    f"<I_B>={IB_exp[k]:+.2f}"
                ),
                va="center",
                fontsize=7,
            )

        ax.set_title(
            f"JT {jt_label}",
            color=JT_COLORS[JT_LABELS.index(jt_label)],
        )
        ax.set_xlabel("Energy above ground state (MHz)")
        ax.set_yticks(range(6))
        ax.set_yticklabels(
            [f"|ψ{k}>" for k in range(6)]
        )
        ax.grid(alpha=0.12)

    fig.suptitle(
        "P1 energy levels at the measured field",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    pdf.savefig(fig)
    save_sim_figure_dm(
        fig,
        timestamp,
        "energy-levels",
    )
    plt.close(fig)



# =============================================================================
# MAIN
# =============================================================================

def main():
    kpl.init_kplotlib()

    timestamp = dm.get_time_stamp()
    file_path = dm.get_file_path(
        __file__,
        timestamp,
        "p1-deer-simulation-52G",
    )

    file_path_str = str(file_path)
    base_no_ext = str(Path(file_path_str).with_suffix(""))
    pdf_path = Path(base_no_ext + ".pdf")

    print("=" * 72)
    print("P1 DEER SIMULATION")
    print("=" * 72)

    B_mag = np.linalg.norm(B_VEC)
    B_hat = B_VEC / B_mag

    print(
        f"B = [{B_VEC[0]:.3f}, "
        f"{B_VEC[1]:.3f}, "
        f"{B_VEC[2]:.3f}] G"
    )
    print(f"|B| = {B_mag:.4f} G")
    print(
        "B_hat = "
        f"[{B_hat[0]:.6f}, "
        f"{B_hat[1]:.6f}, "
        f"{B_hat[2]:.6f}]"
    )

    selected_nv = compute_nv_transitions(
        B_VEC,
        SELECTED_NV_AXIS,
    )

    print("\nSelected NV family:")
    print(f"  {SELECTED_NV_LABEL}")
    print(
        f"  lower = "
        f"{selected_nv['f_lower']/1000:.6f} GHz"
    )
    print(
        f"  upper = "
        f"{selected_nv['f_upper']/1000:.6f} GHz"
    )

    print("\nComputing P1 transitions...")
    freqs, all_transitions, spectra = build_deer_spectrum(
        B_VEC,
        FREQ_RANGE_MHZ,
        CONCENTRATIONS_PPB,
        mw_pulse_us=MW_PULSE_US,
        n_points=N_POINTS,
    )

    significant = get_significant_transitions(
        all_transitions
    )

    print(
        f"Found {len(significant)} significant "
        f"P1 transitions with weight >= "
        f"{SIGNIFICANT_WEIGHT}."
    )

    print("\nSignificant transitions:")
    for row in significant:
        print(
            f"  {row['JT']:<11s} "
            f"{row['freq']:8.3f} MHz  "
            f"weight={row['weight']:.4f}"
        )

    # --------------------------------------------------------
    # Save analyzed simulation data through standard data_manager
    # --------------------------------------------------------

    transition_freq_mhz = np.ascontiguousarray(
        np.asarray(
            [row["freq"] for row in significant],
            dtype=float,
        )
    )
    transition_weight = np.ascontiguousarray(
        np.asarray(
            [row["weight"] for row in significant],
            dtype=float,
        )
    )
    transition_intensity = np.ascontiguousarray(
        np.asarray(
            [row["intensity"] for row in significant],
            dtype=float,
        )
    )
    transition_delta_SB = np.ascontiguousarray(
        np.asarray(
            [row["delta_SB"] for row in significant],
            dtype=float,
        )
    )

    # Store JT family as integer 0..3 for compressed numerical output.
    transition_jt_ind = np.ascontiguousarray(
        np.asarray(
            [
                JT_LABELS.index(row["JT"])
                for row in significant
            ],
            dtype=int,
        )
    )

    concentrations_ppb = np.ascontiguousarray(
        np.asarray(CONCENTRATIONS_PPB, dtype=float)
    )
    spectra_matrix = np.ascontiguousarray(
        np.vstack(
            [
                np.asarray(spectra[c], dtype=float)
                for c in CONCENTRATIONS_PPB
            ]
        )
    )

    sim_data = {
        "timestamp": timestamp,
        "simulation": "P1 DEER 52G full 6x6 Hamiltonian",
        "B_vec_G": np.ascontiguousarray(B_VEC),
        "B_mag_G": float(B_mag),
        "B_hat": np.ascontiguousarray(B_hat),
        "D_NV_MHz": float(D_NV),
        "selected_nv_reference_ghz": float(TARGET_NV_GHZ),
        "selected_nv_axis": np.ascontiguousarray(SELECTED_NV_AXIS),
        "selected_nv_lower_MHz": float(selected_nv["f_lower"]),
        "selected_nv_upper_MHz": float(selected_nv["f_upper"]),
        "mw_pulse_us": float(MW_PULSE_US),
        "freq_range_MHz": np.ascontiguousarray(
            np.asarray(FREQ_RANGE_MHZ, dtype=float)
        ),
        "freqs_MHz": np.ascontiguousarray(freqs),
        "concentrations_ppb": concentrations_ppb,
        "spectra": spectra_matrix,
        "transition_freq_MHz": transition_freq_mhz,
        "transition_weight": transition_weight,
        "transition_intensity": transition_intensity,
        "transition_delta_SB": transition_delta_SB,
        "transition_jt_ind": transition_jt_ind,
        "jt_labels": JT_LABELS,
        "A_par_MHz": float(A_PAR),
        "A_perp_MHz": float(A_PERP),
        "P_par_MHz": float(P_PAR),
        "gamma_e_MHz_per_G": float(GAMMA_E),
        "significant_weight_threshold": float(SIGNIFICANT_WEIGHT),
    }

    keys_to_compress = [
        "B_vec_G",
        "B_hat",
        "selected_nv_axis",
        "freq_range_MHz",
        "freqs_MHz",
        "concentrations_ppb",
        "spectra",
        "transition_freq_MHz",
        "transition_weight",
        "transition_intensity",
        "transition_delta_SB",
        "transition_jt_ind",
    ]

    dm.save_raw_data(
        sim_data,
        file_path,
        keys_to_compress,
    )

    print("\nGenerating combined PDF...")

    with PdfPages(pdf_path) as pdf:
        add_summary_page(
            pdf,
            all_transitions,
            timestamp,
        )

        add_spectrum_page(
            pdf,
            freqs,
            all_transitions,
            spectra,
            timestamp,
        )

        add_transition_table_pages(
            pdf,
            all_transitions,
        )

        add_concentration_page(
            pdf,
            freqs,
            spectra,
            timestamp,
        )

        add_energy_levels_page(pdf, timestamp)


        # PDF metadata
        d = pdf.infodict()
        d["Title"] = "P1 DEER Simulation — 52 G QNami Field"
        d["Subject"] = (
            "P1 transition simulation report"
        )
        d["Author"] = "Saroj Chand"

    print("\nSaved:")
    print(f"  DM data : {file_path}")
    print(f"  PDF     : {pdf_path}")
    print("=" * 72)

    plt.show(block=True)


if __name__ == "__main__":
    main()
