"""
Visualize nuclear-spin ESEEM as *conditional nuclear precession* (spin-1/2).

What you get:
1) M(τ) from the Hahn-echo commutator U_echo(τ) = U_-1 U_0 U_-1† U_0†
2) The standard closed form M(τ) = 1 - 2 k sin^2(ω0 τ/2) sin^2(ω-1 τ/2)
3) Nuclear Bloch-vector precession under the two conditional Hamiltonians
   (m=0 and m=-1), showing the geometric origin of ESEEM.

Units:
- B in Tesla
- hyperfine A in Hz (vector A = [Ax, Ay, Az] in Hz)
- gamma_n_2pi in Hz/T (default: 13C, gamma/2π ≈ 10.705 MHz/T)

If you prefer B in Gauss, set B_G and convert: 1 G = 1e-4 T.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import kplotlib as kpl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch


kpl.init_kplotlib()
# -----------------------------
# Constants / settings
# -----------------------------
gamma_13C_2pi = 10.705e6  # Hz/T (13C)
B_T = np.array([0.0, 0.0, 0.05])  # Tesla (0.05 T = 500 G)
A_Hz = np.array([200e3, 0.0, 40e3])  # hyperfine vector (Hz) in lab/NV frame
tau_us = np.linspace(0.0, 200.0, 1501)  # τ in microseconds
tau_s = tau_us * 1e-6


# ---------- Helpers ----------
def norm(v):
    return float(np.sqrt(np.dot(v, v)))


def unit(v, eps=1e-30):
    n = norm(v)
    return v / max(n, eps)


# Pauli matrices
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


def Omega_vectors(B_T, A_Hz, gamma_n_2pi=gamma_13C_2pi):
    # Ω0  = 2π * ((γ/2π) B) in rad/s
    # Ω-1 = 2π * ((γ/2π) B - A) where A is in Hz
    fB_Hz_vec = gamma_n_2pi * np.asarray(B_T, float)  # Hz
    Om0 = 2 * np.pi * fB_Hz_vec  # rad/s
    Omm1 = 2 * np.pi * (fB_Hz_vec - np.asarray(A_Hz, float))  # rad/s
    return Om0, Omm1


def U_from_Omega(Omega_rad_s, t_s):
    Omega = np.asarray(Omega_rad_s, float)
    w = norm(Omega)
    if w < 1e-30:
        return I2.copy()
    n = Omega / w
    n_dot_sigma = n[0] * sx + n[1] * sy + n[2] * sz
    return np.cos(w * t_s / 2) * I2 - 1j * np.sin(w * t_s / 2) * n_dot_sigma


def U_echo(Om0, Omm1, tau_s):
    U0 = U_from_Omega(Om0, tau_s)
    Um1 = U_from_Omega(Omm1, tau_s)
    return Um1 @ U0 @ Um1.conj().T @ U0.conj().T


def M_trace(Om0, Omm1, tau_array_s):
    Ms = np.zeros_like(tau_array_s, dtype=float)
    for i, tt in enumerate(tau_array_s):
        Ue = U_echo(Om0, Omm1, tt)
        Ms[i] = 0.5 * np.trace(Ue).real
    return Ms


def M_closed_form(Om0, Omm1, tau_array_s):
    w0, w1 = norm(Om0), norm(Omm1)
    n0, n1 = unit(Om0), unit(Omm1)
    k = norm(np.cross(n0, n1)) ** 2
    M = 1.0 - 2.0 * k * (np.sin(w0 * tau_array_s / 2) ** 2) * (
        np.sin(w1 * tau_array_s / 2) ** 2
    )
    return M, k


def rotation_about_axis(n, angle):
    # Rodrigues rotation
    n = unit(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]], float)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ---------- Compute ----------
Om0, Omm1 = Omega_vectors(B_T, A_Hz)
M_num = M_trace(Om0, Omm1, tau_s)
M_cf, k_geom = M_closed_form(Om0, Omm1, tau_s)

n0, n1 = unit(Om0), unit(Omm1)
w0, w1 = norm(Om0), norm(Omm1)

# Example nuclear Bloch trajectories for the schematic
r_init = unit(np.array([1.0, 0.2, 0.1]))
t_arc_us = np.linspace(0, 25, 250)
t_arc_s = t_arc_us * 1e-6
r_arc0 = np.array([rotation_about_axis(n0, w0 * t) @ r_init for t in t_arc_s])
r_arc1 = np.array([rotation_about_axis(n1, w1 * t) @ r_init for t in t_arc_s])

# ---------- (1) Timeline-only ----------
fig1 = plt.figure(figsize=(6.2, 2.6))
ax = fig1.add_subplot(1, 1, 1)
ax.set_title("Hahn echo sequence (NV)")
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

ax.plot([1, 9], [3, 3])
ax.add_patch(Rectangle((2 - 0.08, 2.6), 0.16, 0.8, fill=False))
ax.text(2, 3.55, r"$\pi/2$", ha="center", va="bottom")
ax.add_patch(Rectangle((6 - 0.08, 2.6), 0.16, 0.8, fill=False))
ax.text(6, 3.55, r"$\pi$", ha="center", va="bottom")
ax.text(4, 3.15, r"$\tau$", ha="center")
ax.text(8, 3.15, r"$\tau$", ha="center")
ax.text(1, 3.55, "NV:", ha="right", va="bottom")

ax.plot([2.1, 6], [2.0, 2.0])
ax.plot([6.1, 9], [2.0, 2.0])
ax.text(4.05, 2.15, r"$m=0$", ha="center")
ax.text(7.55, 2.15, r"$m=-1$", ha="center")

ax.text(1.0, 1.2, r"$\Omega_{0}=\gamma_n B$", ha="left")
ax.text(1.0, 0.6, r"$\Omega_{-1}=\gamma_n B - A$", ha="left")
ax.text(1.0, 0.0, rf"$k=|\hat n_0\times \hat n_{-1}|^2={k_geom:.3f}$", ha="left")

fig1.tight_layout()
fig1.savefig("eseem_timeline_only.png", dpi=240, bbox_inches="tight")

# ---------- (2) 3D Bloch-sphere-only ----------
fig2 = plt.figure(figsize=(5.3, 5.1))
ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
ax2.set_title("Conditional nuclear precession")
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)
ax2.set_box_aspect((1, 1, 1))
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

u = np.linspace(0, 2 * np.pi, 34)
v = np.linspace(0, np.pi, 18)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax2.plot_wireframe(xs, ys, zs, linewidth=0.3, rstride=2, cstride=2)

ax2.quiver(0, 0, 0, n0[0], n0[1], n0[2], length=1.0, normalize=True)
ax2.quiver(0, 0, 0, n1[0], n1[1], n1[2], length=1.0, normalize=True)
ax2.text(n0[0], n0[1], n0[2], r"$\hat n_0$", ha="left")
ax2.text(n1[0], n1[1], n1[2], r"$\hat n_{-1}$", ha="left")

ax2.plot(r_arc0[:, 0], r_arc0[:, 1], r_arc0[:, 2], label="precession (m=0)")
ax2.plot(r_arc1[:, 0], r_arc1[:, 1], r_arc1[:, 2], label="precession (m=-1)")
ax2.legend(loc="upper left", fontsize=8)
ax2.view_init(elev=18, azim=40)

fig2.tight_layout()
fig2.savefig("eseem_bloch3d_only.png", dpi=240, bbox_inches="tight")

# ---------- (3) Modulation-only ----------
fig3 = plt.figure(figsize=(6.2, 3.0))
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_title("ESEEM modulation factor")
ax3.plot(tau_us, M_num, label=r"$M(\tau)=\frac{1}{2}\mathrm{Tr}[U_{\mathrm{echo}}]$")
ax3.plot(
    tau_us, M_cf, "--", label=r"$1-2k\sin^2(\omega_0\tau/2)\sin^2(\omega_{-1}\tau/2)$"
)
ax3.set_xlabel(r"$\tau\ (\mu s)$")
ax3.set_ylabel(r"$M(\tau)$")
ax3.legend(fontsize=9)
ax3.grid(True, linewidth=0.4)

fig3.tight_layout()
fig3.savefig("eseem_modulation_only.png", dpi=240, bbox_inches="tight")

print(
    "Saved: eseem_timeline_only.png, eseem_bloch3d_only.png, eseem_modulation_only.png"
)


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def make_spin_echo_template_like_v7(
    out_png="spin_echo_template_like_v7.png",
    out_svg="spin_echo_template_like_v7.svg",
    W=854,
    H=307,
):
    row_labels = ["NV 0", "NV 1", "NV 2", "⋮", "NV n", "Global"]
    n_rows = len(row_labels)

    top_margin = 55
    row_gap = 38
    y_rows = [top_margin + i * row_gap for i in range(n_rows)]
    y_global = y_rows[-1]

    x_left = 70
    x_charge0, x_charge1 = 95, 250
    x_spin0, x_spin1 = 270, 395

    # MW closer to spin-pol (reduced gap)
    x_rf0 = x_spin1 + 12
    rf_width = 285
    x_rf1 = x_rf0 + rf_width

    x_scc0, x_scc1 = 560, 715
    x_ro0, x_ro1 = 735, 855

    green_pulses = [105, 140, None, None, 240, None]  # charge init
    red_pulses = [None] * n_rows  # SCC set after MW

    green_pulse_w = 18
    red_pulse_w = 11
    pulse_h = 28

    C_GREEN_FACE, C_GREEN_EDGE = "#7FBF7B", "#2A8F2A"
    C_RED_FACE, C_RED_EDGE = "#F5A3A3", "#D7191C"
    C_Y_FACE, C_Y_EDGE = "#F7F5B6", "#D8D100"
    C_MW_FACE, C_MW_EDGE = "#B7B0B0", "#7F7A7A"

    y_top = y_rows[0] - 18
    global_h = y_global - y_top

    def add_pulse_from_baseline(
        ax, x, y_base, w, h, face, edge, lw=2, ls="-", alpha=1.0, z=2
    ):
        ax.add_patch(
            Rectangle(
                (x, y_base - h),
                w,
                h,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                linestyle=ls,
                alpha=alpha,
                zorder=z,
            )
        )

    def break_squiggle(ax, x_center, y, gap_w=26, amp=6, lw=1.2):
        # mask baseline behind the squiggle (taller mask = cleaner look)
        ax.add_patch(
            Rectangle(
                (x_center - gap_w / 2, y - amp - 4),
                gap_w,
                2 * amp + 8,
                facecolor="white",
                edgecolor="none",
                zorder=4,
            )
        )
        # symmetric zig-zag (two "teeth")
        xs = [
            x_center - gap_w / 2,
            x_center - gap_w / 4,
            x_center,
            x_center + gap_w / 4,
            x_center + gap_w / 2,
        ]
        ys = [y, y - amp, y + amp, y - amp, y]
        ax.plot(xs, ys, color="black", linewidth=lw, zorder=5)

    def draw_baseline_with_gaps(ax, x0, x1, y, gaps, lw=1.0):
        intervals = []
        for a, b in gaps:
            if b <= x0 or a >= x1:
                continue
            intervals.append((max(x0, a), min(x1, b)))
        intervals.sort()

        merged = []
        for a, b in intervals:
            if not merged or a > merged[-1][1]:
                merged.append([a, b])
            else:
                merged[-1][1] = max(merged[-1][1], b)

        cur = x0
        for a, b in merged:
            if a > cur:
                ax.plot([cur, a], [y, y], color="black", linewidth=lw, zorder=1)
            cur = b
        if cur < x1:
            ax.plot([cur, x1], [y, y], color="black", linewidth=lw, zorder=1)

    # --- MW Hahn echo blocks: π/2 – τ – π – τ – π/2 ---
    mw_w_pi = 32
    mw_w_pi2 = mw_w_pi // 2
    tau_gap = 24

    x_pi2_1 = x_rf0 + 18
    x_pi = x_pi2_1 + mw_w_pi2 + tau_gap
    x_pi2_2 = x_pi + mw_w_pi + tau_gap

    mw_blocks = [
        (x_pi2_1, mw_w_pi2, r"$\pi/2$"),
        (x_pi, mw_w_pi, r"$\pi$"),
        (x_pi2_2, mw_w_pi2, r"$\pi/2$"),
    ]

    # SCC starts after last π/2 ends
    last_mw_end = x_pi2_2 + mw_w_pi2
    margin = 15
    x_scc_start = max(x_scc0 + 5, last_mw_end + margin)

    # SCC stagger matches GREEN offsets exactly
    green_non_none = [g for g in green_pulses if g is not None]
    g0 = green_non_none[0]
    green_offsets = [(g - g0) if g is not None else None for g in green_pulses]
    for i, off in enumerate(green_offsets):
        if off is None:
            continue
        red_x = x_scc_start + off
        red_x = min(x_scc1 - red_pulse_w - 5, red_x)
        red_pulses[i] = red_x

    # Break locations (second is slightly narrower so it looks cleaner)
    brk1 = x_charge1 - 45
    brk2 = x_scc1 - 50
    brk_gap_w1 = 28
    brk_gap_w2 = 24
    brk_intervals = [
        (brk1 - brk_gap_w1 / 2, brk1 + brk_gap_w1 / 2),
        (brk2 - brk_gap_w2 / 2, brk2 + brk_gap_w2 / 2),
    ]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Baseline gaps: remove inside global blocks + under pulses + under break
    global_gaps = [(x_spin0, x_spin1), (x_ro0, x_ro1)]
    for xb, ww, _ in mw_blocks:
        global_gaps.append((xb, xb + ww))
    global_gaps.extend(brk_intervals)

    for i, y in enumerate(y_rows):
        gaps = list(global_gaps)
        if green_pulses[i] is not None:
            gaps.append(
                (green_pulses[i], green_pulses[i] + green_pulse_w)
            )  # correct width
        if red_pulses[i] is not None:
            gaps.append((red_pulses[i], red_pulses[i] + red_pulse_w))
        draw_baseline_with_gaps(ax, x_left, x_ro1, y, gaps, lw=1.0)
        ax.text(18, y + 5, row_labels[i], fontsize=16, ha="left", va="center")

    # Titles
    ax.text(
        (x_charge0 + x_charge1) / 2,
        15,
        "Charge init.",
        fontsize=15,
        ha="center",
        va="center",
    )

    # MW label centered over actual MW pulse span (fixes offset)
    mw_label_x = (x_pi2_1 + last_mw_end) / 2
    ax.text(mw_label_x, 15, "Spin echo", fontsize=15, ha="center", va="center")

    ax.text((x_scc0 + x_scc1) / 2, 15, "SCC", fontsize=15, ha="center", va="center")

    # Green pulses (anchored to baseline)
    for y, xp in zip(y_rows, green_pulses):
        if xp is None:
            continue
        add_pulse_from_baseline(
            ax, xp, y, green_pulse_w, pulse_h, C_GREEN_FACE, C_GREEN_EDGE, lw=2, z=3
        )

    # Break after charge
    for y in y_rows:
        break_squiggle(ax, brk1, y, gap_w=brk_gap_w1, amp=6, lw=1.2)

    # 589 nm spin pol (global)
    spin_rect = Rectangle(
        (x_spin0, y_top),
        x_spin1 - x_spin0,
        global_h,
        facecolor=C_Y_FACE,
        edgecolor=C_Y_EDGE,
        linewidth=2,
        zorder=2,
        alpha=0.6,
    )
    ax.add_patch(spin_rect)
    ax.text(
        (x_spin0 + x_spin1) / 2,
        (y_top + y_global) / 2,
        "spin pol",
        fontsize=15,
        rotation=90,
        ha="center",
        va="center",
        zorder=3,
    )

    # MW pulses (global)
    for xb, ww, lab in mw_blocks:
        add_pulse_from_baseline(
            ax, xb, y_global, ww, global_h, C_MW_FACE, C_MW_EDGE, lw=2, alpha=0.55, z=2
        )
        ax.text(
            xb + ww / 2,
            (y_top + y_global) / 2,
            lab,
            fontsize=15,
            rotation=90,
            ha="center",
            va="center",
            zorder=3,
        )

    # SCC red pulses (anchored to baseline)
    for y, xp in zip(y_rows, red_pulses):
        if xp is None:
            continue
        add_pulse_from_baseline(
            ax, xp, y, red_pulse_w, pulse_h, C_RED_FACE, C_RED_EDGE, lw=2, z=3
        )

    # Break after SCC (cleaner)
    for y in y_rows:
        break_squiggle(ax, brk2, y, gap_w=brk_gap_w2, amp=6, lw=1.2)

    # Readout (global, open right)
    ro_fill = Rectangle(
        (x_ro0, y_top),
        x_ro1 - x_ro0,
        global_h,
        facecolor=C_Y_FACE,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    ax.add_patch(ro_fill)
    ax.plot([x_ro0, x_ro0], [y_top, y_global], color=C_Y_EDGE, linewidth=2, zorder=3)
    ax.plot([x_ro0, x_ro1], [y_top, y_top], color=C_Y_EDGE, linewidth=2, zorder=3)
    ax.plot([x_ro0, x_ro1], [y_global, y_global], color=C_Y_EDGE, linewidth=2, zorder=3)

    ax.text(
        (x_ro0 + x_ro1) / 2,
        (y_top + y_global) / 2,
        "Readout",
        fontsize=15,
        rotation=90,
        ha="center",
        va="center",
        zorder=3,
    )

    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    # plt.close(fig)


if __name__ == "__main__":
    make_spin_echo_template_like_v7()
