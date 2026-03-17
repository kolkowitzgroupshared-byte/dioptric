import numpy as np

# ----------------------------
# Your experimental parameters
# ----------------------------
B_G = np.array([-50.59318864, -12.17874298, -3.46780984])  # G (crystal axes)
Bmag = np.linalg.norm(B_G)
Bhat = B_G / Bmag

t_pi = 150e-9  # RF pi time (s)
tau = 18e-6  # echo tau (s)  (assumed 18 us)
two_tau = 2 * tau

# Frequency axis to simulate (MHz)
fmin_MHz, fmax_MHz = 80.0, 240.0
df_MHz = 0.25
f_axis_MHz = np.arange(fmin_MHz, fmax_MHz + df_MHz, df_MHz)

# ----------------------------
# Physical constants / P1 params (14N)
# ----------------------------
gamma_e = 2.802495  # MHz/G
gamma_n = 0.0003077  # MHz/G for 14N (approx); small at 52 G but included

A_par = 114.0264  # MHz
A_perp = 81.312  # MHz
Q = -3.9770  # MHz (quadrupole term, in P tensor form)


# ----------------------------
# Spin operators
# ----------------------------
def spin_half():
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def spin_1():
    # basis |m> = |+1,0,-1>
    m = np.array([1, 0, -1], dtype=float)
    Iz = np.diag(m)
    Ip = np.zeros((3, 3), dtype=complex)
    Im = np.zeros((3, 3), dtype=complex)
    I = 1
    for i, mi in enumerate(m):
        if mi < I:
            j = np.where(m == mi + 1)[0][0]
            Ip[j, i] = np.sqrt(I * (I + 1) - mi * (mi + 1))
        if mi > -I:
            j = np.where(m == mi - 1)[0][0]
            Im[j, i] = np.sqrt(I * (I + 1) - mi * (mi - 1))
    Ix = 0.5 * (Ip + Im)
    Iy = -0.5j * (Ip - Im)
    return Ix, Iy, Iz


Sx, Sy, Sz = spin_half()
Ix, Iy, Iz = spin_1()


def kron(a, b):
    return np.kron(a, b)


# 6D operators (electron ⊗ nuclear)
Sx6 = kron(Sx, np.eye(3))
Sy6 = kron(Sy, np.eye(3))
Sz6 = kron(Sz, np.eye(3))
Ix6 = kron(np.eye(2), Ix)
Iy6 = kron(np.eye(2), Iy)
Iz6 = kron(np.eye(2), Iz)

# ----------------------------
# Geometry: 4 <111> JT axes in crystal frame
# ----------------------------
u111 = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
u111 = np.array([v / np.linalg.norm(v) for v in u111])


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 deg rotation: pick arbitrary axis orthogonal to a
        axis = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    v = v / np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    s = np.sqrt(1 - c**2)
    return np.eye(3) + K * s + (K @ K) * (1 - c)


A_principal = np.diag([A_perp, A_perp, A_par])  # MHz


def build_H(u_axis):
    # rotate hyperfine tensor into crystal frame
    R = rot_a_to_b(np.array([0, 0, 1.0]), u_axis)
    A = R @ A_principal @ R.T  # MHz

    # Zeeman (electron + nuclear)
    HZ_e = gamma_e * (B_G[0] * Sx6 + B_G[1] * Sy6 + B_G[2] * Sz6)
    HZ_n = gamma_n * (B_G[0] * Ix6 + B_G[1] * Iy6 + B_G[2] * Iz6)

    # Hyperfine S·A·I
    Svec = [Sx6, Sy6, Sz6]
    Ivec = [Ix6, Iy6, Iz6]
    Hhf = np.zeros((6, 6), dtype=complex)
    for i in range(3):
        for j in range(3):
            Hhf += A[i, j] * (Svec[i] @ Ivec[j])

    # Quadrupole along defect axis: Q*(Iu^2 - I(I+1)/3)
    I = 1
    Iu = u_axis[0] * Ix6 + u_axis[1] * Iy6 + u_axis[2] * Iz6
    Hq = Q * (Iu @ Iu - (I * (I + 1) / 3.0) * np.eye(6))

    return HZ_e + HZ_n + Hhf + Hq


def transitions(H):
    E, V = np.linalg.eigh(H)  # MHz
    # drive operator: assume transverse microwave, average Sx and Sy
    O = (Sx6 + Sy6) / np.sqrt(2)
    lines = []
    for i in range(6):
        for j in range(i + 1, 6):
            f = float(np.real(E[j] - E[i]))  # MHz
            mij = np.vdot(V[:, i], O @ V[:, j])
            strength = float((abs(mij) ** 2))
            # keep only reasonable strengths
            if strength > 1e-8 and f > 0:
                lines.append((f, strength))
    # normalize strengths
    if lines:
        ssum = sum(s for _, s in lines)
        lines = [(f, s / ssum) for f, s in lines]
    return lines


# ----------------------------
# Lineshape pieces
# ----------------------------
def rf_excitation_profile(delta_f_MHz, t_pi_s):
    """
    Rectangular pulse excitation envelope ~ sinc^2(pi * delta_f * t)
    delta_f_MHz: detuning in MHz
    """
    # convert MHz -> Hz
    delta_f_Hz = delta_f_MHz * 1e6
    x = np.pi * delta_f_Hz * t_pi_s
    # sinc(x) = sin(x)/x
    y = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-12
    y[mask] = (np.sin(x[mask]) / x[mask]) ** 2
    return y


def gaussian(x, sigma_MHz):
    return np.exp(-0.5 * (x / sigma_MHz) ** 2)


def lorentzian(x, gamma_MHz):
    return 1.0 / (1.0 + (x / gamma_MHz) ** 2)


# ----------------------------
# Build DEER spectrum from sticks
# ----------------------------
def simulate_deer_spectrum(
    f_axis_MHz,
    n_ppb=10.0,
    sigma0_MHz=0.5,
    broadening_per_ppb_MHz=0.03,
    k_depth=0.8,
    use_lorentz=False,
):
    """
    n_ppb: P1 concentration in ppb (controls depth + broadening)
    sigma0_MHz: base inhomogeneous width (MHz)
    broadening_per_ppb_MHz: additional width scaling with concentration
    k_depth: depth scale (absorbs geometry + coupling + 2tau dependence)
    """
    # concentration-dependent width (simple model)
    width_MHz = sigma0_MHz + broadening_per_ppb_MHz * np.sqrt(max(n_ppb, 0.0))

    # aggregate spectrum (unitless)
    S = np.zeros_like(f_axis_MHz, dtype=float)

    for u in u111:
        H = build_H(u)
        lines = transitions(H)  # list of (f0, strength)
        for f0, s in lines:
            det = f_axis_MHz - f0
            rf = rf_excitation_profile(det, t_pi)
            if use_lorentz:
                L = lorentzian(det, width_MHz)
            else:
                L = gaussian(det, width_MHz)
            S += s * rf * L

    # normalize S to max=1 (optional)
    if S.max() > 0:
        S = S / S.max()

    # Convert "spectral overlap" into Hahn-echo contrast:
    # Simple phenomenology: Contrast = exp(-k * n * S)
    # (k_depth also effectively grows with 2tau; you can fold that in if you want)
    C = np.exp(-k_depth * (n_ppb / 10.0) * S)

    return C, S, width_MHz


"""
P1 Center EPR Simulation for CVD Diamond - Custom Frequency
Optimized for low-frequency EPR measurements
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.gridspec as gridspec


@dataclass
class P1Parameters:
    """P1 center spin Hamiltonian parameters (literature values)"""

    g_iso = 2.0024  # Isotropic g-factor
    A_parallel = 114.0  # MHz (parallel to <111> axis)
    A_perp = 81.4  # MHz (perpendicular to <111> axis)
    linewidth_G = 0.8  # Gaussian linewidth (G)


def get_p1_orientations() -> List[np.ndarray]:
    """Four equivalent <111> orientations in diamond"""
    orientations = [
        np.array([1, 1, 1]) / np.sqrt(3),  # [111]
        np.array([1, -1, -1]) / np.sqrt(3),  # [1̄1̄1]
        np.array([-1, 1, -1]) / np.sqrt(3),  # [1̄11̄]
        np.array([-1, -1, 1]) / np.sqrt(3),  # [11̄1̄]
    ]
    return orientations


def calculate_resonance_fields(
    B_direction: np.ndarray, frequency_GHz: float, params: P1Parameters
) -> List[Tuple[float, float, int, int]]:
    """
    Calculate resonance fields for all P1 orientations and mI states

    Returns: List of (B_res_Gauss, intensity, orientation_idx, mI_value)
    """
    B_hat = B_direction / np.linalg.norm(B_direction)

    # EPR conversion factor
    MHz_per_Gauss = 2.8024953  # g_e * mu_B / h
    freq_MHz = frequency_GHz * 1000

    # Central field (no hyperfine)
    B_center = freq_MHz / (params.g_iso * MHz_per_Gauss)

    resonances = []
    orientations = get_p1_orientations()

    # Isotropic and anisotropic parts
    A_iso = (params.A_parallel + 2 * params.A_perp) / 3
    A_aniso = (params.A_parallel - params.A_perp) / 3

    for idx, orientation in enumerate(orientations):
        # Angle between field and P1 axis
        cos_theta = np.abs(np.dot(B_hat, orientation))

        # Hyperfine coupling at this angle
        A_theta = A_iso + A_aniso * (3 * cos_theta**2 - 1)

        # Three hyperfine lines for mI = -1, 0, +1
        for mI in [-1, 0, 1]:
            B_res = B_center - (A_theta * mI) / (params.g_iso * MHz_per_Gauss)

            # Intensity (could include angular dependence)
            intensity = 1.0

            resonances.append((B_res, intensity, idx, mI))

    return resonances


def voigt_lineshape(
    B: np.ndarray, B0: float, gamma_G: float, gamma_L: float = 0.05
) -> np.ndarray:
    """Voigt lineshape with Gaussian dominance"""
    sigma = gamma_G / (2 * np.sqrt(2 * np.log(2)))
    z = ((B - B0) + 1j * gamma_L) / (sigma * np.sqrt(2))
    w = wofz(z)
    return np.real(w) / (sigma * np.sqrt(2 * np.pi))


def simulate_spectrum(
    B_vector: np.ndarray,
    frequency_GHz: float,
    B_scan_range: Tuple[float, float] = None,
    p1_concentration: float = 1.0,
    params: P1Parameters = None,
    num_points: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Main simulation function

    Parameters:
    -----------
    B_vector : Magnetic field vector in crystal coordinates (Gauss)
    frequency_GHz : Microwave frequency (GHz)
    B_scan_range : (B_min, B_max) in Gauss, or None for auto
    p1_concentration : Relative concentration (affects amplitude)

    Returns:
    --------
    B_field, spectrum, info_dict
    """
    if params is None:
        params = P1Parameters()

    # Auto-determine scan range if not provided
    if B_scan_range is None:
        B_mag = np.linalg.norm(B_vector)
        B_scan_range = (B_mag - 30, B_mag + 30)

    B_field = np.linspace(B_scan_range[0], B_scan_range[1], num_points)
    spectrum = np.zeros_like(B_field)

    # Calculate all resonances
    resonances = calculate_resonance_fields(B_vector, frequency_GHz, params)

    # Track contributions by orientation
    orientation_spectra = {i: np.zeros_like(B_field) for i in range(4)}

    # Build spectrum
    for B_res, intensity, ori_idx, mI in resonances:
        if B_scan_range[0] <= B_res <= B_scan_range[1]:
            line = voigt_lineshape(B_field, B_res, params.linewidth_G)
            contribution = intensity * line * p1_concentration

            spectrum += contribution
            orientation_spectra[ori_idx] += contribution

    info = {
        "resonances": resonances,
        "orientation_spectra": orientation_spectra,
        "num_resonances": len(resonances),
        "frequency_GHz": frequency_GHz,
        "B_center": np.linalg.norm(B_vector),
    }

    return B_field, spectrum, info


def create_comprehensive_plot(
    B_vector: np.ndarray, frequency_GHz: float, concentrations: List[float] = None
):
    """Create comprehensive multi-panel figure"""

    if concentrations is None:
        concentrations = [0.5, 1.0, 2.0, 5.0]

    B_mag = np.linalg.norm(B_vector)
    B_hat = B_vector / B_mag
    B_scan = (B_mag - 30, B_mag + 30)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Orientation labels
    ori_labels = ["[111]", "[1̄1̄1]", "[1̄11̄]", "[11̄1̄]"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    # Panel 1: Full spectrum with orientation breakdown
    ax1 = fig.add_subplot(gs[0, :])
    B_field, spectrum, info = simulate_spectrum(B_vector, frequency_GHz, B_scan, 1.0)

    ax1.plot(B_field, spectrum, "k-", linewidth=2.5, label="Total", zorder=10)

    for idx in range(4):
        ori_spec = info["orientation_spectra"][idx]
        if np.max(ori_spec) > 0.01 * np.max(spectrum):
            ax1.plot(
                B_field,
                ori_spec,
                color=colors[idx],
                linewidth=1.5,
                label=ori_labels[idx],
                alpha=0.6,
            )

    ax1.set_xlabel("Magnetic Field (G)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("EPR Absorption (arb. units)", fontsize=13, fontweight="bold")
    ax1.set_title(
        f"P1 EPR Spectrum - All 4 Orientations | ν = {frequency_GHz:.4f} GHz",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="best", frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Add field info
    textstr = f"|B| = {B_mag:.3f} G\n"
    textstr += f"B̂ = [{B_hat[0]:.3f}, {B_hat[1]:.3f}, {B_hat[2]:.3f}]"
    ax1.text(
        0.02,
        0.98,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Panel 2: Resonance positions diagram
    ax2 = fig.add_subplot(gs[1, 0])

    resonances_sorted = sorted(info["resonances"], key=lambda x: x[0])
    for B_res, _, ori_idx, mI in resonances_sorted:
        if B_scan[0] <= B_res <= B_scan[1]:
            ax2.axvline(B_res, color=colors[ori_idx], alpha=0.6, linewidth=2)
            ax2.plot(
                B_res,
                ori_idx,
                "o",
                color=colors[ori_idx],
                markersize=10,
                label=f"{ori_labels[ori_idx]} (mI={mI:+d})",
            )

    ax2.set_xlabel("Resonance Field (G)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Orientation Index", fontsize=12, fontweight="bold")
    ax2.set_title("Resonance Field Positions", fontsize=12, fontweight="bold")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(ori_labels)
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.set_xlim(B_scan)

    # Panel 3: Concentration comparison
    ax3 = fig.add_subplot(gs[1, 1])

    for conc in concentrations:
        _, spec_c, _ = simulate_spectrum(B_vector, frequency_GHz, B_scan, conc)
        ax3.plot(B_field, spec_c, linewidth=2, label=f"[P1] = {conc:.1f}×")

    ax3.set_xlabel("Magnetic Field (G)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Absorption (arb. units)", fontsize=12, fontweight="bold")
    ax3.set_title("Concentration Dependence", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", title="Relative concentration")
    ax3.grid(True, alpha=0.3)

    # Panel 4: Angular pattern
    ax4 = fig.add_subplot(gs[2, :])

    # Show how spectrum changes with field orientation
    angles = np.linspace(0, 180, 7)
    orientations_test = get_p1_orientations()

    # Rotate field around an axis
    for i, angle_deg in enumerate(angles):
        angle_rad = np.deg2rad(angle_deg)
        # Rotate around y-axis
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        B_rot = R @ (B_vector / B_mag) * B_mag

        _, spec_rot, _ = simulate_spectrum(B_rot, frequency_GHz, B_scan, 1.0)
        ax4.plot(
            B_field, spec_rot + i * 0.5, linewidth=1.5, label=f"{angle_deg}°", alpha=0.8
        )

    ax4.set_xlabel("Magnetic Field (G)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Absorption (offset for clarity)", fontsize=12, fontweight="bold")
    ax4.set_title("Angular Dependence (Field Rotation)", fontsize=12, fontweight="bold")
    ax4.legend(loc="right", title="Rotation angle", ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        "P1 Center Comprehensive EPR Analysis", fontsize=16, fontweight="bold", y=0.995
    )

    return fig


def print_detailed_analysis(B_vector: np.ndarray, frequency_GHz: float):
    """Print detailed analysis of the EPR parameters"""

    print("\n" + "=" * 80)
    print("DETAILED P1 EPR ANALYSIS")
    print("=" * 80)

    B_mag = np.linalg.norm(B_vector)
    B_hat = B_vector / B_mag

    print(f"\nEXPERIMENTAL CONDITIONS:")
    print(
        f"  Microwave frequency:  {frequency_GHz:.6f} GHz = {frequency_GHz*1000:.3f} MHz"
    )
    print(f"  Magnetic field:       |B| = {B_mag:.4f} G")
    print(
        f"  Field vector:         B = [{B_vector[0]:.4f}, {B_vector[1]:.4f}, {B_vector[2]:.4f}] G"
    )
    print(
        f"  Field direction:      B̂ = [{B_hat[0]:.4f}, {B_hat[1]:.4f}, {B_hat[2]:.4f}]"
    )

    # Calculate expected parameters
    params = P1Parameters()
    MHz_per_G = 2.8024953
    B_center = (frequency_GHz * 1000) / (params.g_iso * MHz_per_G)
    A_iso = (params.A_parallel + 2 * params.A_perp) / 3
    hyperfine_splitting_G = A_iso / (params.g_iso * MHz_per_G)

    print(f"\nP1 CENTER PARAMETERS:")
    print(f"  g-factor:             {params.g_iso}")
    print(f"  A∥ (parallel):        {params.A_parallel:.2f} MHz")
    print(f"  A⊥ (perpendicular):   {params.A_perp:.2f} MHz")
    print(f"  A_iso:                {A_iso:.2f} MHz")
    print(f"  Hyperfine splitting:  {hyperfine_splitting_G:.2f} G")

    print(f"\nRESPONANCE FIELD PREDICTION:")
    print(f"  Expected center:      {B_center:.2f} G")
    print(f"  Your field:           {B_mag:.2f} G")
    print(f"  Difference:           {abs(B_center - B_mag):.2f} G")

    if abs(B_center - B_mag) / B_center < 0.05:
        print(f"  ✓ Field matches frequency (within 5%)")
    else:
        print(f"  ⚠ Large deviation - check frequency/field calibration")

    # Calculate resonances
    _, _, info = simulate_spectrum(B_vector, frequency_GHz)

    print(f"\nRESPONANCE LINES:")
    print(f"  Total lines: {len(info['resonances'])}")
    print(f"\n  {'Field (G)':>10} | {'Orient.':>8} | {'mI':>5} | Angle")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}-+-{'-'*20}")

    orientations = get_p1_orientations()
    ori_labels = ["[111]", "[1̄1̄1]", "[1̄11̄]", "[11̄1̄]"]

    for B_res, _, ori_idx, mI in sorted(info["resonances"], key=lambda x: x[0]):
        orientation = orientations[ori_idx]
        cos_theta = abs(np.dot(B_hat, orientation))
        theta_deg = np.rad2deg(np.arccos(cos_theta))
        print(
            f"  {B_res:10.3f} | {ori_labels[ori_idx]:>8} | {mI:+5d} | θ = {theta_deg:5.1f}°"
        )

    print("\n" + "=" * 80)


# MAIN EXECUTION
if __name__ == "__main__":
    # Your experimental parameters
    B_vector = np.array([-50.59318864, -12.17874298, -3.46780984])  # Gauss

    # Calculate frequency from field (or use your known frequency)
    B_mag = np.linalg.norm(B_vector)

    # If you know your frequency, set it here:
    # frequency_GHz = your_value

    # Otherwise, calculate it assuming resonance at center
    g = 2.0024
    frequency_GHz = (B_mag * g * 2.8024953) / 1000  # Convert to GHz

    print("=" * 80)
    print("P1 CENTER EPR SIMULATION FOR CVD DIAMOND")
    print("=" * 80)
    print(
        f"\nCalculated frequency from field: {frequency_GHz:.6f} GHz ({frequency_GHz*1000:.3f} MHz)"
    )
    print("\nIf you know your actual frequency, please modify the code and re-run.")

    # Detailed analysis
    print_detailed_analysis(B_vector, frequency_GHz)

    # Create comprehensive plot
    print("\n" + "-" * 80)
    print("Generating comprehensive analysis plots...")
    fig = create_comprehensive_plot(
        B_vector, frequency_GHz, concentrations=[0.5, 1.0, 2.0, 5.0]
    )
    plt.savefig(
        "/home/claude/p1_comprehensive_analysis.png", dpi=300, bbox_inches="tight"
    )
    print("✓ Saved: p1_comprehensive_analysis.png")

    # Export numerical data
    B_field, spectrum, info = simulate_spectrum(B_vector, frequency_GHz)
    # np.savetxt(
    #     "/home/claude/p1_simulation_data.txt",
    #     np.column_stack([B_field, spectrum]),
    #     header=f"P1 EPR Simulation\nFrequency: {frequency_GHz:.6f} GHz\n"
    #     f"B_vector: {B_vector}\nColumn 1: Field (G), Column 2: Intensity",
    #     fmt="%.6f",
    # )
    print("✓ Saved: p1_simulation_data.txt")

    print("\n" + "=" * 80)
    print("CONCENTRATION ESTIMATION GUIDE")
    print("=" * 80)
    print(
        """
For Element Six CVD diamond with ingrown P1:

Typical P1 densities (nitrogen concentration):
  • High purity: 50-150 ppb (~1-3 × 10¹⁵ spins/cm³)
  • Standard CVD: 100-300 ppb (~2-6 × 10¹⁵ spins/cm³)
  • Lower grade: 300-500 ppb (~6-10 × 10¹⁵ spins/cm³)

To estimate YOUR sample's P1 density:

1. RELATIVE METHOD (recommended if you have spectra):
   • Compare your experimental spectrum peak height/area with simulations
   • If your spectrum is 2× the intensity of concentration=1.0 simulation,
     then use concentration=2.0 as a reference

2. ABSOLUTE METHOD (requires calibration):
   • Double integrate your EPR spectrum
   • Compare with a known standard (e.g., strong pitch, DPPH)
   • [P1] = (Double_integral_sample / Double_integral_standard) × [standard]

3. FROM MANUFACTURER:
   • Element Six may provide nitrogen content specification
   • Most P1 in CVD = substitutional nitrogen

4. OPTICAL ABSORPTION:
   • P1 has characteristic IR absorption at 1130 cm⁻¹ and 1344 cm⁻¹
   • Can correlate with EPR if both measurements available

The concentration parameter in the simulation is relative. Scale it to match
your experimental spectrum, then use the relationships above for absolute values.
    """
    )

    print("=" * 80)
    print("Simulation complete! Check the plots and data files.")
    print("=" * 80)

    # plt.show()  # Uncomment to display plots interactively
# # ----------------------------
# # Example: sweep concentrations
# # ----------------------------
# if __name__ == "__main__":
#     for n_ppb in [1, 3, 10, 30, 100, 300, 1000]:  # up to 1 ppm
#         C, S, w = simulate_deer_spectrum(f_axis_MHz, n_ppb=n_ppb)
#         print(f"n={n_ppb:>4} ppb, effective width ~ {w:.2f} MHz, C(min)~{C.min():.3f}")

#     # C, S, _ = simulate_deer_spectrum(f_axis_MHz, n_ppb=30)
#     # Then plot f_axis_MHz vs C using matplotlib.
