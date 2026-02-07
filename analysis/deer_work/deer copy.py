#!/usr/bin/env python3
"""
P1 Center DEER Spectrum Simulator
==================================
Simulates Double Electron-Electron Resonance (DEER) spectra of P1 centers
(substitutional nitrogen) in diamond, as detected by a single NV center.

Physics:
  - Full 6x6 numerical diagonalization of the P1 Hamiltonian (S=1/2 ⊗ I=1)
  - All 4 Jahn-Teller axes
  - All 4 NV center orientations
  - Concentration-dependent dipolar linewidth
  - Transition intensities including low-field electron-nuclear mixing

NEW (DEER contrast model, more explicit):
  - Adds an explicit *echo-contrast* model for dip depth vs tau based on a
    quasi-static bath of resonant P1 spins coupled via dipolar Ising terms.
  - Models the dip depth per spectral line as:
        C_k(conc, tau) = 1 - exp[-(tau / T_DEER,k(conc))^p]   (stretched exp)
    multiplied by:
        (i)   JT population (1/4),
        (ii)  the P1 transition "DEER weight" w_k = intensity * sigma,
        (iii) the finite spectral excitation probability of the P1 pi pulse.
  - The pulse excitation probability is modeled as a Gaussian in frequency with
    FWHM ≈ 0.89 / t_pi (rectangular-pulse bandwidth proxy).
  - This removes the purely ad-hoc "0.05" depth scaling and replaces it with a
    tau-dependent depth that saturates with increasing tau.

Important note:
  - "sigma" here is computed using the *lab Sz* for the P1 electron.
    For stricter NV-axis physics you would use S·n_NV in the dipolar term.
    (Kept as-is to avoid changing your existing transition code.)

Reference: Degen et al., Nature Communications 12, 3470 (2021)
           Cox, Newton & Baker, J. Phys.: Condens. Matter 6, 551 (1994)
           Nir-Arad et al., PCCP 26, 27633 (2024)
"""

import numpy as np
from scipy.linalg import eigh

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
from utils import kplotlib as kpl

warnings.filterwarnings("ignore", category=RuntimeWarning)
kpl.init_kplotlib()

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
GAMMA_E = 2.8025  # Electron gyromagnetic ratio [MHz/G] (g ≈ 2.0024)
GAMMA_N14 = 3.077e-4  # 14N nuclear gyromagnetic ratio [MHz/G]
A_PAR = 114.0  # Hyperfine A_∥ [MHz]
A_PERP = 81.3  # Hyperfine A_⊥ [MHz]
P_PAR = -3.97  # Quadrupole P_∥ [MHz]
D_NV = 2878.5  # NV zero-field splitting [MHz]
CARBON_DENSITY = 1.764e23  # Carbon atoms per cm³ in diamond

# =============================================================================
# CRYSTALLOGRAPHIC AXES
# =============================================================================
JT_AXES = np.array(
    [
        [1, 1, 1],  # JT-A
        [-1, -1, 1],  # JT-B
        [-1, 1, -1],  # JT-C
        [1, -1, -1],  # JT-D
    ]
) / np.sqrt(3)
JT_LABELS = ["A [111]", "B [-1-11]", "C [-11-1]", "D [1-1-1]"]

NV_AXES = np.array(
    [
        [1, 1, 1],  # NV-1
        [-1, -1, 1],  # NV-2
        [-1, 1, -1],  # NV-3
        [1, -1, -1],  # NV-4
    ]
) / np.sqrt(3)
NV_LABELS = ["NV [111]", "NV [-1-11]", "NV [-11-1]", "NV [1-1-1]"]


def sinc(x):
    # normalized sinc: sin(pi x)/(pi x)
    return np.sinc(x)


def p1_pi_pulse_inversion_prob(detuning_mhz, omega_rabi_mhz, t_pi_us):
    """
    Probability that a driven two-level transition is inverted by a resonant pi-pulse,
    as a function of detuning.
    For a square pulse: P = (Ω^2/Ω_eff^2) * sin^2(pi * Ω_eff * t / 2)
    where Ω_eff = sqrt(Ω^2 + Δ^2).
    Here detuning in MHz, Ω in MHz, t in microseconds.
    """
    delta = detuning_mhz
    Omega = omega_rabi_mhz
    Omega_eff = np.sqrt(Omega**2 + delta**2)

    # For t = t_pi, on resonance delta=0 gives sin^2(pi/2)=1
    arg = np.pi * Omega_eff * t_pi_us / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (Omega**2 / (Omega_eff**2)) * (np.sin(arg) ** 2)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(P, 0.0, 1.0)


def deer_contrast_prefactor(
    tau_us, T2_nv_us=1000.0, T2_deer_us=800.0, n_nv=2.0, n_deer=2.0
):
    """
    Simple, explicit envelope model:
      - NV echo reference envelope ~ exp(-(2tau/T2_nv)^n_nv)
      - Extra DEER-induced decay amplitude ~ (1 - exp(-(2tau/T2_deer)^n_deer))
    This is *not* the full cluster expansion, but it makes τ-dependence explicit.
    """
    two_tau = 2.0 * tau_us
    nv_env = np.exp(-((two_tau / T2_nv_us) ** n_nv))
    deer_amp = 1.0 - np.exp(-((two_tau / T2_deer_us) ** n_deer))
    return nv_env * deer_amp


# =============================================================================
# SPIN OPERATORS
# =============================================================================
def build_spin_operators():
    """
    Build the 6x6 spin operators for S=1/2 (electron) ⊗ I=1 (14N nucleus).
    Basis: |↑,+1>, |↑,0>, |↑,-1>, |↓,+1>, |↓,0>, |↓,-1>
    """
    # S = 1/2
    sx_half = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy_half = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz_half = np.array([[1, 0], [0, -1]], dtype=complex) / 2
    I2 = np.eye(2, dtype=complex)

    # I = 1, basis: |+1>, |0>, |-1>
    sq2 = np.sqrt(2)
    ix_one = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / sq2
    iy_one = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / sq2
    iz_one = np.diag([1.0, 0.0, -1.0]).astype(complex)
    I3 = np.eye(3, dtype=complex)

    # 6x6 via Kronecker products
    Sx = np.kron(sx_half, I3)
    Sy = np.kron(sy_half, I3)
    Sz = np.kron(sz_half, I3)
    Ix = np.kron(I2, ix_one)
    Iy = np.kron(I2, iy_one)
    Iz = np.kron(I2, iz_one)

    return [Sx, Sy, Sz], [Ix, Iy, Iz]


def build_nv_spin1_operators():
    """Build spin-1 operators for the NV ground state. Basis: |+1>, |0>, |-1>."""
    sq2 = np.sqrt(2)
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / sq2
    Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / sq2
    Sz = np.diag([1.0, 0.0, -1.0]).astype(complex)
    return Sx, Sy, Sz


# =============================================================================
# P1 HAMILTONIAN
# =============================================================================
def build_p1_hamiltonian(B_vec, jt_axis, S_ops, I_ops):
    """
    Build full 6x6 P1 center Hamiltonian.
    H = γ_e B·S − γ_N B·I + S·A·I + P_∥[(n·I)² − I(I+1)/3]
    """
    Sx, Sy, Sz = S_ops
    Ix, Iy, Iz = I_ops
    n = jt_axis
    dim = 6

    H_eZ = GAMMA_E * (B_vec[0] * Sx + B_vec[1] * Sy + B_vec[2] * Sz)
    H_nZ = -GAMMA_N14 * (B_vec[0] * Ix + B_vec[1] * Iy + B_vec[2] * Iz)

    S_dot_I = Sx @ Ix + Sy @ Iy + Sz @ Iz
    n_dot_S = n[0] * Sx + n[1] * Sy + n[2] * Sz
    n_dot_I = n[0] * Ix + n[1] * Iy + n[2] * Iz
    H_HF = A_PERP * S_dot_I + (A_PAR - A_PERP) * (n_dot_S @ n_dot_I)

    H_Q = P_PAR * (n_dot_I @ n_dot_I - (2.0 / 3.0) * np.eye(dim, dtype=complex))

    return H_eZ + H_nZ + H_HF + H_Q


def diagonalize_p1(B_vec, jt_axis, S_ops, I_ops):
    """Diagonalize P1 Hamiltonian. Returns sorted eigenvalues [MHz] and eigenvectors."""
    H = build_p1_hamiltonian(B_vec, jt_axis, S_ops, I_ops)
    return eigh(H)


def compute_p1_transitions(B_vec, jt_axis, S_ops, I_ops, intensity_threshold=1e-4):
    """
    Compute all electron-spin transitions of a P1 center.
    Returns list of dicts with freq, intensity, sigma (effective spin flip), deer_weight.
    """
    Sx, Sy, Sz = S_ops
    evals, evecs = diagonalize_p1(B_vec, jt_axis, S_ops, I_ops)

    B_hat = B_vec / np.linalg.norm(B_vec)

    # Two orthonormal vectors perpendicular to B (MW polarization directions)
    if abs(B_hat[2]) < 0.9:
        perp1 = np.cross(B_hat, [0, 0, 1])
    else:
        perp1 = np.cross(B_hat, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(B_hat, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    S_perp1 = perp1[0] * Sx + perp1[1] * Sy + perp1[2] * Sz
    S_perp2 = perp2[0] * Sx + perp2[1] * Sy + perp2[2] * Sz

    transitions = []
    for i in range(6):
        for j in range(i + 1, 6):
            freq = evals[j] - evals[i]
            M1 = np.abs(evecs[:, j].conj() @ S_perp1 @ evecs[:, i]) ** 2
            M2 = np.abs(evecs[:, j].conj() @ S_perp2 @ evecs[:, i]) ** 2
            intensity = float(np.real(M1 + M2))

            Sz_j = float(np.real(evecs[:, j].conj() @ Sz @ evecs[:, j]))
            Sz_i = float(np.real(evecs[:, i].conj() @ Sz @ evecs[:, i]))
            sigma = abs(Sz_j - Sz_i)

            if intensity > intensity_threshold:
                transitions.append(
                    {
                        "freq": float(freq),
                        "intensity": float(intensity),
                        "sigma": float(sigma),
                        "deer_weight": float(intensity * sigma),
                        "states": (i, j),
                    }
                )
    return transitions


# =============================================================================
# NV CENTER
# =============================================================================
def compute_nv_transitions(B_vec, nv_axis):
    """Compute NV transition frequencies for a given orientation."""
    Sx_nv, Sy_nv, Sz_nv = build_nv_spin1_operators()

    B_par = np.dot(B_vec, nv_axis)
    B_perp_vec = B_vec - B_par * nv_axis
    B_perp = np.linalg.norm(B_perp_vec)

    z_nv = nv_axis
    if B_perp > 1e-10:
        x_nv = B_perp_vec / B_perp
    else:
        if abs(z_nv[2]) < 0.9:
            x_nv = np.cross(z_nv, [0, 0, 1])
        else:
            x_nv = np.cross(z_nv, [1, 0, 0])
        x_nv /= np.linalg.norm(x_nv)
    y_nv = np.cross(z_nv, x_nv)

    Bx_nv = np.dot(B_vec, x_nv)
    By_nv = np.dot(B_vec, y_nv)
    Bz_nv = np.dot(B_vec, z_nv)

    H_nv = D_NV * (Sz_nv @ Sz_nv) + GAMMA_E * (
        Bx_nv * Sx_nv + By_nv * Sy_nv + Bz_nv * Sz_nv
    )
    evals, evecs = eigh(H_nv)

    overlaps_0 = np.abs(evecs[1, :]) ** 2
    idx_0 = np.argmax(overlaps_0)
    other_idx = [k for k in range(3) if k != idx_0]
    f_transitions = sorted([np.abs(evals[k] - evals[idx_0]) for k in other_idx])

    return {
        "B_par": float(B_par),
        "B_perp": float(B_perp),
        "f_lower": float(f_transitions[0]),
        "f_upper": float(f_transitions[1]),
        "evals": evals,
    }


# =============================================================================
# DEER CONTRAST MODEL (NEW)
# =============================================================================
def deer_depth_stretched_exp(tau_us, T_deer_us, p=1.0):
    """
    Frequency-domain DEER dip depth from a quasi-static bath model:
      C(tau) = 1 - exp[-(tau/T_DEER)^p]
    - p=1 is exponential (Lorentzian-like noise), p=2 is Gaussian-like.
    """
    tau_us = float(max(tau_us, 0.0))
    T_deer_us = float(max(T_deer_us, 1e-12))
    p = float(max(p, 1e-6))
    return 1.0 - np.exp(-((tau_us / T_deer_us) ** p))


def estimate_Tdeer_us(conc_ppb, Tref_us=250.0, cref_ppb=75.0, alpha=1.0):
    """
    Simple scaling for the DEER timescale with concentration:
      1/T_DEER ∝ n ∝ conc
      => T_DEER(conc) = Tref * (cref/conc)^alpha

    Tref_us is a tunable "calibration knob" that sets overall dip depth vs tau.
    Use your data to calibrate Tref_us at a known concentration.
    """
    conc_ppb = float(max(conc_ppb, 1e-12))
    return float(Tref_us * (cref_ppb / conc_ppb) ** alpha)


def pulse_excitation_probability(df_MHz, t_pi_us, model="gaussian"):
    """
    Probability that the DEER pi-pulse excites a transition detuned by df.
    Uses an approximate bandwidth from a rectangular pulse:
        FWHM ~ 0.89 / t_pi_us   (MHz)

    We approximate the excitation profile as a Gaussian in frequency for simplicity.
    """
    t_pi_us = float(max(t_pi_us, 1e-12))
    fwhm = 0.89 / t_pi_us  # MHz
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # convert FWHM to std dev

    if model == "gaussian":
        return float(np.exp(-0.5 * (df_MHz / sigma) ** 2))
    # fallback: hard top-hat within FWHM/2
    return float(1.0 if abs(df_MHz) <= 0.5 * fwhm else 0.0)


# =============================================================================
# DEER SPECTRUM CONSTRUCTION
# =============================================================================
def gaussian_lineshape(f, f0, sigma):
    """Gaussian lineshape (sigma is the std dev, NOT FWHM)."""
    return np.exp(-0.5 * ((f - f0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def fwhm_to_sigma(fwhm):
    """Convert FWHM to Gaussian sigma."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def concentration_to_linewidth(conc_ppb, mw_pulse_us=4.0):
    """
    Estimate total P1 linewidth (FWHM, MHz).
    Dipolar: ~4.5 MHz at 1 ppm, scales linearly.
    MW pulse bandwidth: ~0.89/t_pulse (sinc).
    Intrinsic T2*: ~50 µs → ~6.4 kHz FWHM = 0.0064 MHz (negligible).
    """
    gamma_dipolar = 4.5e-3 * conc_ppb  # MHz
    gamma_mw = 0.89 / mw_pulse_us  # MHz
    gamma_T2star = 0.0064  # MHz (from T2* ~ 50 µs)
    return np.sqrt(gamma_dipolar**2 + gamma_mw**2 + gamma_T2star**2)


def build_deer_spectrum(
    B_vec,
    freq_range,
    concentrations_ppb,
    mw_pulse_us=4.0,
    n_points=4000,
    tau_us=18.0,
    deer_stretch_p=1.0,
    Tref_us=250.0,
    cref_ppb=75.0,
    T_alpha=1.0,
    excitation_model="gaussian",
    omega_rabi_mhz=None,  # <-- NEW (P1 Rabi in MHz)
    T2_nv_us=992.0,  # <-- NEW (from paper ~0.992 ms)  :contentReference[oaicite:3]{index=3}
    T2_deer_us=780.0,  # <-- NEW (from paper ~0.76–0.80 ms) :contentReference[oaicite:4]{index=4}
    A0=0.15,  # <-- NEW global contrast scale (fit this)
):
    """
    Build DEER spectra for all 4 JT axes at various P1 concentrations.

    NEW: dip depth is tau-dependent and saturating via a stretched-exponential model.

    Args:
        tau_us: total interaction time used in DEER (echo half-time is tau; total 2*tau,
                but in simple quasi-static models the relevant scale is O(tau).
                Keep as a tunable knob. (Set tau_us=18 for your 18 us echo tau.)
        deer_stretch_p: stretch exponent p in 1-exp(-(tau/T)^p)
        Tref_us, cref_ppb, T_alpha: set T_DEER scaling with concentration
        excitation_model: "gaussian" or "tophat" for pi-pulse excitation probability

    Returns:
        freqs, all_transitions (dict by JT label), spectra (dict by conc)
    """
    S_ops, I_ops = build_spin_operators()
    f_min, f_max = freq_range
    freqs = np.linspace(f_min, f_max, n_points)
    if omega_rabi_mhz is None:
        # A square pi pulse: Omega = 1/(2 t_pi) in cycles/us = MHz
        # because pi rotation occurs when Omega * t_pi = 1/2 (in cycles)
        omega_rabi_mhz = 1.0 / (2.0 * mw_pulse_us)
    # --- Compute all transitions ---
    all_transitions = {}
    for jt_idx, (jt_ax, jt_label) in enumerate(zip(JT_AXES, JT_LABELS)):
        trans = compute_p1_transitions(
            B_vec, jt_ax, S_ops, I_ops, intensity_threshold=1e-4
        )
        for t in trans:
            t["jt_label"] = jt_label
            t["jt_idx"] = jt_idx
        all_transitions[jt_label] = trans

    # --- Build spectrum for each concentration ---
    spectra = {}
    for conc in concentrations_ppb:
        # linewidth of each spectral line
        fwhm = concentration_to_linewidth(conc, mw_pulse_us)
        sig_line = fwhm_to_sigma(fwhm)
        pref = deer_contrast_prefactor(
            tau_us=tau_us,
            T2_nv_us=T2_nv_us,
            T2_deer_us=T2_deer_us,
            n_nv=2.0,
            n_deer=2.0,
        )

        # tau-dependent saturation depth (global scale for this concentration)
        T_deer = estimate_Tdeer_us(
            conc, Tref_us=Tref_us, cref_ppb=cref_ppb, alpha=T_alpha
        )
        depth_sat = deer_depth_stretched_exp(
            tau_us=tau_us, T_deer_us=T_deer, p=deer_stretch_p
        )

        spectrum = np.ones_like(freqs)

        for jt_label, trans_list in all_transitions.items():
            jt_weight = 0.25  # Equal population of JT axes

            for t in trans_list:
                # Transition-specific weight from Hamiltonian mixing + MW matrix elements
                w = float(
                    t["deer_weight"]
                )  # intensity * sigma (dimensionless-ish proxy)

                # Excitation probability of the DEER pi pulse at each frequency point.
                # We fold this into the *lineshape amplitude* by multiplying the dip depth
                # by P_exc(f - f0).
                # (Alternative: convolve; this is a good fast approximation.)
                #
                # Choose a "base depth" that saturates with tau and scales with w and JT pop.
                base_depth = jt_weight * w * depth_sat

                # Keep physical: depth cannot exceed 1, and you often see a few %.
                # You can tune with Tref_us (and optionally a global factor below).
                # If you want an extra knob, change global_scale from 1.0 to e.g. 0.2.
                global_scale = 1.0
                base_depth *= global_scale

                # Apply finite pulse bandwidth by reducing depth away from resonance.
                # We'll do this point-by-point by multiplying the peak profile.
                peak = gaussian_lineshape(freqs, t["freq"], sig_line)

                # Normalize peak so its maximum is 1 (so base_depth is the on-resonance dip)
                peak_max = 1.0 / (sig_line * np.sqrt(2 * np.pi))
                if peak_max <= 0:
                    continue
                peak_norm = peak / peak_max  # max=1

                # Extra: multiply by excitation probability at each detuning
                # (This effectively narrows the dip if your pi pulse is narrow.)
                df = freqs - t["freq"]
                if excitation_model == "gaussian":
                    # vectorized excitation probability
                    fwhm_exc = 0.89 / float(max(mw_pulse_us, 1e-12))
                    sig_exc = fwhm_to_sigma(fwhm_exc)
                    P_exc = np.exp(-0.5 * (df / sig_exc) ** 2)
                else:
                    # tophat
                    fwhm_exc = 0.89 / float(max(mw_pulse_us, 1e-12))
                    P_exc = (np.abs(df) <= 0.5 * fwhm_exc).astype(float)

                dip_profile = base_depth * peak_norm * P_exc

                spectrum -= dip_profile

        spectra[conc] = np.clip(spectrum, 0, 1)

    return freqs, all_transitions, spectra


# =============================================================================
# TEXT OUTPUT
# =============================================================================
def print_field_analysis(B_vec):
    B_mag = np.linalg.norm(B_vec)
    B_hat = B_vec / B_mag
    print("=" * 70)
    print("MAGNETIC FIELD ANALYSIS")
    print("=" * 70)
    print(f"  B = [{B_vec[0]:.4f}, {B_vec[1]:.4f}, {B_vec[2]:.4f}] G")
    print(f"  |B| = {B_mag:.4f} G")
    print(f"  B_hat = [{B_hat[0]:.6f}, {B_hat[1]:.6f}, {B_hat[2]:.6f}]")
    print(f"  Electron Zeeman: gamma_e*|B| = {GAMMA_E * B_mag:.2f} MHz")
    print(f"  A_par = {A_PAR} MHz  -->  gamma_e|B|/A_par = {GAMMA_E*B_mag/A_PAR:.3f}")
    print(f"  --> STRONG electron-nuclear mixing regime")
    print()
    print("  Angle between B and each JT axis:")
    for jt_ax, label in zip(JT_AXES, JT_LABELS):
        cos_theta = np.dot(B_hat, jt_ax)
        theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        print(f"    JT-{label}: theta = {theta:.1f} deg  (cos = {cos_theta:.4f})")
    print()
    print("  NV center transition frequencies:")
    for nv_ax, nv_label in zip(NV_AXES, NV_LABELS):
        info = compute_nv_transitions(B_vec, nv_ax)
        print(
            f"    {nv_label}:  B_par={info['B_par']:.2f} G  B_perp={info['B_perp']:.2f} G"
            f"  f_lower={info['f_lower']:.1f} MHz  f_upper={info['f_upper']:.1f} MHz"
        )
    print()


def print_transition_table(all_transitions, B_vec):
    B_mag = np.linalg.norm(B_vec)
    larmor = GAMMA_E * B_mag
    print("=" * 70)
    print(f"P1 TRANSITIONS  (bare Larmor = {larmor:.2f} MHz)")
    print("=" * 70)
    for jt_label in JT_LABELS:
        trans = sorted(all_transitions[jt_label], key=lambda t: t["freq"])
        print(f"\n  JT-{jt_label}:")
        print(
            f"  {'Freq [MHz]':>12}  {'Intensity':>10}  {'sigma':>10}  {'DEER wt':>10}  {'States':>8}"
        )
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        for t in trans:
            if t["deer_weight"] > 0.001:
                print(
                    f"  {t['freq']:12.4f}  {t['intensity']:10.6f}  "
                    f"{t['sigma']:10.6f}  {t['deer_weight']:10.6f}  "
                    f"  {t['states'][0]}-{t['states'][1]}"
                )


def print_concentration_table():
    print("\n" + "=" * 70)
    print("P1 CONCENTRATION REFERENCE (Element Six CVD)")
    print("=" * 70)
    data = [
        (1, "EL-grade (ultra-pure)"),
        (5, "EL-grade (spec <5 ppb)"),
        (10, "Custom low-N"),
        (50, "Custom (Degen-like)"),
        (75, "Degen et al. 2021"),
        (100, "Moderate CVD"),
        (200, "Optical-grade"),
        (500, "Standard CVD"),
        (1000, "DNV-B1 (~800 ppb)"),
        (10000, "HPHT type Ib"),
    ]
    print(
        f"  {'[N] ppb':>8}  {'ppm':>7}  {'n (cm-3)':>12}  {'r_nn (nm)':>10}  {'Gamma_dip':>10}  {'Grade'}"
    )
    print(f"  {'-'*8}  {'-'*7}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*25}")
    for c, grade in data:
        n = c * 1e-9 * CARBON_DENSITY
        r = 0.554 / (n ** (1.0 / 3.0)) * 1e7  # nm
        gd = 4.5e-3 * c
        print(f"  {c:8d}  {c/1000:7.3f}  {n:12.3e}  {r:10.1f}  {gd:10.4f}  {grade}")
    print()


# =============================================================================
# PLOTTING
# =============================================================================
def plot_all(
    freqs, all_transitions, spectra, B_vec, concentrations, save_dir, mw_pulse_us=4.0
):
    """Generate all 4 publication-quality figures."""
    B_mag = np.linalg.norm(B_vec)
    larmor = GAMMA_E * B_mag
    jt_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    # Reference concentration for single-spectrum plots
    ref_conc = 5 if 5 in spectra else min(concentrations, key=lambda c: abs(c - 75))

    # ---- FIGURE 1: Stick spectrum + broadened reference spectrum ----
    fig1, (ax_stick, ax_spec) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    for jt_idx, (jt_label, color) in enumerate(zip(JT_LABELS, jt_colors)):
        for t in all_transitions[jt_label]:
            if t["deer_weight"] > 0.003:
                ax_stick.vlines(
                    t["freq"],
                    0,
                    t["deer_weight"],
                    colors=color,
                    linewidth=1.5,
                    alpha=0.8,
                )
                ax_stick.plot(
                    t["freq"], t["deer_weight"], "o", color=color, markersize=4
                )

    ax_stick.axvline(larmor, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_stick.set_ylabel("DEER weight\n(intensity x sigma)", fontsize=15)
    ax_stick.set_title(
        f"P1 DEER Spectrum  |  B = [{B_vec[0]:.1f}, {B_vec[1]:.1f}, "
        f"{B_vec[2]:.1f}] G  (|B| = {B_mag:.1f} G)",
        fontsize=15,
    )
    legend_els = [
        Line2D([0], [0], color=c, marker="o", ls="-", ms=5, label=f"JT-{l}")
        for c, l in zip(jt_colors, JT_LABELS)
    ]
    legend_els.append(Line2D([0], [0], color="gray", ls="--", label="Larmor freq."))
    ax_stick.legend(handles=legend_els, loc="upper right", fontsize=9, ncol=2)
    ax_stick.set_ylim(bottom=0)

    ax_spec.plot(freqs, spectra[ref_conc], "k-", lw=1.2)
    ax_spec.fill_between(freqs, spectra[ref_conc], 1, alpha=0.1, color="steelblue")
    ax_spec.axvline(larmor, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_spec.set_xlabel("Frequency (MHz)", fontsize=15)
    ax_spec.set_ylabel(r"DEER signal ($F_{|m_s=0\rangle}$)", fontsize=15)
    ymin_spec = max(0.5, min(spectra[ref_conc]) - 0.02)
    ax_spec.set_ylim(ymin_spec, 1.02)
    ax_spec.text(
        0.02,
        0.06,
        f"[P1] = {ref_conc} ppb  (MW pulse = {mw_pulse_us:.1f} us)",
        transform=ax_spec.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    plt.tight_layout()
    path1 = f"{save_dir}/deer_stick_and_spectrum.png"
    fig1.savefig(path1, dpi=200, bbox_inches="tight")

    # ---- FIGURE 2: Concentration comparison ----
    n_conc = len(concentrations)
    fig2, axes2 = plt.subplots(
        n_conc, 1, figsize=(8, 2.5 * n_conc), sharex=True, gridspec_kw={"hspace": 0}
    )
    if n_conc == 1:
        axes2 = [axes2]
    cmap = plt.cm.viridis
    for idx, (conc, ax) in enumerate(zip(concentrations, axes2)):
        c_color = cmap(idx / max(n_conc - 1, 1))
        ax.plot(freqs, spectra[conc], color=c_color, lw=1.2)
        ax.fill_between(freqs, spectra[conc], 1, alpha=0.15, color=c_color)
        ax.axvline(larmor, color="gray", ls="--", lw=0.6, alpha=0.4)
        gamma = concentration_to_linewidth(conc, mw_pulse_us)
        ax.text(
            0.02,
            0.15,
            f"[P1] = {conc} ppb  (FWHM = {gamma:.2f} MHz)",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.set_ylabel("Signal", fontsize=15)
        ymin = max(0, min(spectra[conc]) - 0.02)
        ax.set_ylim(ymin, 1.02)
    axes2[-1].set_xlabel("Frequency (MHz)", fontsize=15)
    axes2[0].set_title(
        f"DEER Spectra vs P1 Concentration  |  "
        f"B = [{B_vec[0]:.1f}, {B_vec[1]:.1f}, {B_vec[2]:.1f}] G",
        fontsize=15,
    )
    plt.tight_layout()
    path2 = f"{save_dir}/deer_concentration_comparison.png"
    fig2.savefig(path2, dpi=200, bbox_inches="tight")

    # ---- FIGURE 3: Energy levels per JT axis ----
    S_ops, I_ops = build_spin_operators()
    Sz = S_ops[2]
    Iz = I_ops[2]

    fig3, axes3 = plt.subplots(1, 4, figsize=(12, 7))
    for jt_idx, (jt_ax, jt_label, ax, color) in enumerate(
        zip(JT_AXES, JT_LABELS, axes3, jt_colors)
    ):
        evals, evecs = diagonalize_p1(B_vec, jt_ax, S_ops, I_ops)
        Sz_exp = [
            float(np.real(evecs[:, k].conj() @ Sz @ evecs[:, k])) for k in range(6)
        ]
        Iz_exp = [
            float(np.real(evecs[:, k].conj() @ Iz @ evecs[:, k])) for k in range(6)
        ]

        for k in range(6):
            c_bar = "crimson" if Sz_exp[k] > 0 else "royalblue"
            alpha = 0.3 + 0.7 * min(abs(Sz_exp[k]) * 2, 1.0)
            ax.barh(
                k,
                evals[k],
                height=0.6,
                color=c_bar,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.text(
                evals[k] + 2,
                k,
                f"{evals[k]:.1f}\n<Sz>={Sz_exp[k]:.2f}\n<Iz>={Iz_exp[k]:.2f}",
                va="center",
                fontsize=6.5,
            )
        ax.set_xlabel("Energy (MHz)", fontsize=15)
        ax.set_title(f"JT-{jt_label}", fontsize=15, color=color)
        ax.set_yticks(range(6))
        ax.set_yticklabels([f"|psi_{k}>" for k in range(6)], fontsize=15)

    axes3[0].set_ylabel("Eigenstate", fontsize=15)
    fig3.suptitle(
        f"P1 Energy Levels  |  |B| = {B_mag:.1f} G",
        fontsize=15,
        y=1.01,
    )
    plt.tight_layout()
    path3 = f"{save_dir}/deer_energy_levels.png"
    fig3.savefig(path3, dpi=200, bbox_inches="tight")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # === USER CONFIGURATION ===
    B_vec = np.array([-50.59318864, -12.17874298, -3.46780984])  # Gauss
    concentrations = [5, 25, 75, 100, 200, 500, 1000]  # ppb

    # NOTE: In your earlier messages you said "tau 18us sit at revival".
    # Use tau_us=18.0 here for the DEER contrast model.
    tau_us = 18.0

    # Your DEER pi time for P1: 150 ns -> 0.15 us
    # But mw_pulse_us in this script is used as the *DEER pulse duration* that sets bandwidth.
    # So if you truly use 150 ns pi pulse, set mw_pulse_us = 0.15.
    mw_pulse_us = 0.5  # microseconds

    freq_range = (10, 300)  # MHz
    n_points = 4000
    save_dir = r"analysis/deer_work"

    # Contrast model tuning knobs:
    # Tref_us sets the overall depth scale at cref_ppb for the chosen tau.
    # If dips are too deep/shallow, tune Tref_us.
    deer_stretch_p = 1.0  # try 1.0 or 2.0
    Tref_us = 250.0  # sets depth vs tau at 75 ppb
    cref_ppb = 75.0
    T_alpha = 1.0  # T_DEER ∝ 1/conc

    print("\n" + "=" * 70)
    print("  P1 CENTER DEER SPECTRUM SIMULATION")
    print("=" * 70 + "\n")

    print_field_analysis(B_vec)
    print_concentration_table()

    print("Computing P1 transitions and DEER spectra...")
    freqs, all_transitions, spectra = build_deer_spectrum(
        B_vec,
        freq_range,
        concentrations,
        mw_pulse_us=mw_pulse_us,
        n_points=n_points,
        tau_us=tau_us,
        deer_stretch_p=deer_stretch_p,
        Tref_us=Tref_us,
        cref_ppb=cref_ppb,
        T_alpha=T_alpha,
        excitation_model="gaussian",
    )

    print_transition_table(all_transitions, B_vec)

    # Summary
    print("\n" + "=" * 70)
    print("TRANSITION SUMMARY")
    print("=" * 70)
    for jt_label in JT_LABELS:
        trans = all_transitions[jt_label]
        sig = [t for t in trans if t["deer_weight"] > 0.005]
        print(f"  JT-{jt_label}: {len(sig)} significant transitions")
        for t in sorted(sig, key=lambda x: -x["deer_weight"])[:3]:
            print(f"    f = {t['freq']:.2f} MHz   DEER wt = {t['deer_weight']:.4f}")

    # Plots
    print("\nGenerating figures...")
    plot_all(
        freqs, all_transitions, spectra, B_vec, concentrations, save_dir, mw_pulse_us
    )


if __name__ == "__main__":
    main()
    plt.show(block=True)
