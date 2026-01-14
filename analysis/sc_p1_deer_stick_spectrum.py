import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# USER INPUTS
# ----------------------------
B_vec_G = np.array(
    [-48.67047318, -32.07615947, 22.49657427], dtype=float
)  # [100],[010],[001] basis
observed = np.array([109, 120, 157, 168, 203, 211, 248, 263, 293, 311], dtype=float)

fmin, fmax = 10.0, 360.0  # MHz
gaussian_sigma_MHz = 0.2
intensity_thresh = 1e-5

# P1 (substitutional N, 14N) parameters (MHz)
A_par = 114.0264
A_perp = 81.312
P_par = -3.9770

gamma_e = 2.802495  # MHz/G
gamma_n = 0.0003077  # MHz/G (14N) small


# ----------------------------
# SPIN OPERATORS
# ----------------------------
def spin_matrices_s_half():
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    return Sx, Sy, Sz


def spin_matrices_I1():
    Ix = (1 / np.sqrt(2)) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
    Iy = (1 / np.sqrt(2)) * np.array(
        [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex
    )
    Iz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
    return Ix, Iy, Iz


Sx, Sy, Sz = spin_matrices_s_half()
Ix, Iy, Iz = spin_matrices_I1()

kron = np.kron
Sx6, Sy6, Sz6 = kron(Sx, np.eye(3)), kron(Sy, np.eye(3)), kron(Sz, np.eye(3))
Ix6, Iy6, Iz6 = kron(np.eye(2), Ix), kron(np.eye(2), Iy), kron(np.eye(2), Iz)


# ----------------------------
# GEOMETRY
# ----------------------------
def orthonormal_frame_from_z(z):
    z = np.array(z, dtype=float)
    z = z / np.linalg.norm(z)
    a = np.array([0, 0, 1.0])
    if abs(np.dot(a, z)) > 0.9:
        a = np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, z])


# Four possible JT axes (tetrahedral N–C bonds) in [100],[010],[001] basis
jt_axes = [
    np.array([1, 1, 1], float),
    np.array([1, -1, -1], float),
    np.array([-1, 1, -1], float),
    np.array([-1, -1, 1], float),
]
jt_axes = [a / np.linalg.norm(a) for a in jt_axes]
jt_labels = ["[111]", "[1-1-1]", "[-11-1]", "[-1-11]"]


# ----------------------------
# HAMILTONIAN
# ----------------------------
def p1_hamiltonian(B_vec_G, jt_axis):
    # Hyperfine tensor A in lab frame (principal axis = JT axis)
    R = orthonormal_frame_from_z(jt_axis)
    A0 = np.diag([A_perp, A_perp, A_par])
    A = R @ A0 @ R.T

    # Quadrupole along JT axis: P_par*(I_n^2 - I(I+1)/3)
    n = np.array(jt_axis, dtype=float)
    n /= np.linalg.norm(n)
    I_n = n[0] * Ix6 + n[1] * Iy6 + n[2] * Iz6
    I = 1
    HQ = P_par * (I_n @ I_n - (I * (I + 1) / 3) * np.eye(6))

    # Zeeman terms
    Bx, By, Bz = B_vec_G
    HZ = gamma_e * (Bx * Sx6 + By * Sy6 + Bz * Sz6) - gamma_n * (
        Bx * Ix6 + By * Iy6 + Bz * Iz6
    )

    # Hyperfine S·A·I
    S_ops = [Sx6, Sy6, Sz6]
    I_ops = [Ix6, Iy6, Iz6]
    Hhf = np.zeros((6, 6), dtype=complex)
    for a in range(3):
        for b in range(3):
            Hhf += A[a, b] * (S_ops[a] @ I_ops[b])

    return HZ + Hhf + HQ


def compute_lines(B_vec_G):
    Bhat = B_vec_G / np.linalg.norm(B_vec_G)
    S_B = Bhat[0] * Sx6 + Bhat[1] * Sy6 + Bhat[2] * Sz6

    # crude MW-driving operator (you can swap this if your B1 polarization differs)
    O = Sx6 + 1j * Sy6

    lines = []
    for axis_id, axis in enumerate(jt_axes):
        H = p1_hamiltonian(B_vec_G, axis)
        evals, evecs = np.linalg.eigh(H)

        # precompute expectation values used for labeling
        n = axis / np.linalg.norm(axis)
        I_n = n[0] * Ix6 + n[1] * Iy6 + n[2] * Iz6

        ms = np.array(
            [np.real(np.vdot(evecs[:, k], S_B @ evecs[:, k])) for k in range(6)]
        )
        mi = np.array(
            [np.real(np.vdot(evecs[:, k], I_n @ evecs[:, k])) for k in range(6)]
        )

        for i in range(6):
            for j in range(i + 1, 6):
                f = float(np.real(evals[j] - evals[i]))
                if not (fmin <= f <= fmax):
                    continue
                amp = float(abs(np.vdot(evecs[:, i], O @ evecs[:, j])) ** 2)
                if amp < intensity_thresh:
                    continue

                lines.append(
                    {
                        "f": f,
                        "amp": amp,
                        "axis_id": axis_id,
                        "mi_avg": 0.5
                        * float(mi[i] + mi[j]),  # ~ -1,0,+1 when weakly mixed
                        "dms": float(ms[j] - ms[i]),
                        "dmi": float(mi[j] - mi[i]),
                    }
                )

    lines.sort(key=lambda d: d["f"])
    return lines


lines = compute_lines(B_vec_G)

print(f"|B| = {np.linalg.norm(B_vec_G):.3f} G")
print("Top predicted lines (by strength):")
for d in sorted(lines, key=lambda x: -x["amp"])[:15]:
    print(
        f"  {d['f']:7.2f} MHz  amp={d['amp']:.3g}  JT={jt_labels[d['axis_id']]}  <I_JT>~{d['mi_avg']:+.2f}"
    )


# ----------------------------
# MATCH OBSERVED PEAKS
# ----------------------------
def match_observed(fobs, lines, window=12.0, topk=5):
    cand = [d for d in lines if abs(d["f"] - fobs) <= window]
    cand.sort(key=lambda d: (abs(d["f"] - fobs), -d["amp"]))
    return cand[:topk]


print("\nObserved peak assignments (candidates):")
for fobs in observed:
    cand = match_observed(fobs, lines, window=12.0, topk=4)
    print(f"\nObs {fobs:.1f} MHz:")
    if not cand:
        print(
            "  (no predicted lines within window) -> check B calibration or widen window"
        )
        continue
    for d in cand:
        print(
            f"  pred {d['f']:7.2f} (Δ={d['f']-fobs:+6.2f})  amp={d['amp']:.3g}  JT={jt_labels[d['axis_id']]}  <I_JT>~{d['mi_avg']:+.2f}"
        )

# ----------------------------
# PLOT
# ----------------------------
grid = np.linspace(fmin, fmax, 5000)
spec = np.zeros_like(grid)
for d in lines:
    spec += d["amp"] * np.exp(-((grid - d["f"]) ** 2) / (2 * gaussian_sigma_MHz**2))

plt.figure()
for d in lines:
    plt.vlines(d["f"], 0, d["amp"], linewidth=1)

if spec.max() > 0:
    plt.plot(grid, spec / spec.max())

plt.scatter(observed, np.full_like(observed, 0.05), marker="x")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Relative transition strength (arb.)")
plt.title(
    f"P1 predicted transitions (B={B_vec_G} G; |B|={np.linalg.norm(B_vec_G):.2f} G)"
)
plt.xlim(fmin, fmax)
plt.tight_layout()
# plt.savefig("p1_spectrum_labeled.png", dpi=300)
plt.show()
