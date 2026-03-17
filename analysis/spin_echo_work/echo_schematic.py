import numpy as np
import matplotlib.pyplot as plt


# ---------- Bloch-sphere helpers ----------
def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [1, 0, 0],
            [0, c, -s],
            [0, s, c],
        ]
    )


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ]
    )


def draw_sphere(ax, n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, linewidth=0.5, alpha=0.35)


def draw_axes(ax, L=1.2):
    # coordinate axes
    ax.plot([0, L], [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, L], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, 0], [0, L], linewidth=1)
    ax.text(L, 0, 0, "x", fontsize=10)
    ax.text(0, L, 0, "y", fontsize=10)
    ax.text(0, 0, L, "z", fontsize=10)


def draw_vector(ax, r):
    r = np.asarray(r, float)
    ax.quiver(0, 0, 0, r[0], r[1], r[2], arrow_length_ratio=0.12, linewidth=2)


def arc_on_equator(phi0, phi1, n=80):
    phis = np.linspace(phi0, phi1, n)
    return np.c_[np.cos(phis), np.sin(phis), np.zeros_like(phis)]


def plot_panel(ax, r, title="", eq_arc=None):
    draw_sphere(ax)
    draw_axes(ax)
    draw_vector(ax, r)

    if eq_arc is not None:
        ax.plot(eq_arc[:, 0], eq_arc[:, 1], eq_arc[:, 2], linewidth=2)

    ax.set_title(title, pad=8)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=35)


# ---------- Spin-echo "cartoon" sequence on Bloch sphere ----------
# Choose a phase accumulated during each free-precession interval tau.
# For purely static detuning: phi2 == phi1 and echo refocuses perfectly.
phi1 = np.deg2rad(70)  # phase accumulated during first tau
phi2 = phi1  # set != phi1 to illustrate imperfect refocus / noise

# Start in ms=0 -> +z on Bloch sphere
r0 = np.array([0, 0, 1.0])

# π/2 about x: bring state to equator (coherence)
r1 = Rx(np.pi / 2) @ r0

# Free evolution τ: precession about z by +phi1
r2 = Rz(phi1) @ r1

# π about x: refocus (mirror on equator)
r3 = Rx(np.pi) @ r2

# Free evolution τ: precession about z by +phi2
r4 = Rz(phi2) @ r3

# Final π/2 about x: map coherence back to population
r5 = Rx(np.pi / 2) @ r4


# For drawing equator arcs, extract the azimuth phases of r1, r2, r3, r4
def azimuth(r):
    return np.arctan2(r[1], r[0])


phi_r1 = azimuth(r1)
phi_r2 = azimuth(r2)
phi_r3 = azimuth(r3)
phi_r4 = azimuth(r4)

# ---------- Make multi-panel schematic ----------
fig = plt.figure(figsize=(14, 6))
axs = [
    fig.add_subplot(2, 3, 1, projection="3d"),
    fig.add_subplot(2, 3, 2, projection="3d"),
    fig.add_subplot(2, 3, 3, projection="3d"),
    fig.add_subplot(2, 3, 4, projection="3d"),
    fig.add_subplot(2, 3, 5, projection="3d"),
    fig.add_subplot(2, 3, 6, projection="3d"),
]

plot_panel(axs[0], r0, "Init: +z (ms=0)")
plot_panel(axs[1], r1, r"After $\pi/2_x$")
plot_panel(axs[2], r2, r"After $\tau$ (precess)", eq_arc=arc_on_equator(phi_r1, phi_r2))
plot_panel(axs[3], r3, r"After $\pi_x$")
plot_panel(axs[4], r4, r"After $\tau$ (precess)", eq_arc=arc_on_equator(phi_r3, phi_r4))
plot_panel(axs[5], r5, r"After $\pi/2_x$ (echo readout)")

fig.suptitle(
    "Spin Echo on the Bloch Sphere (static detuning refocused; nuclear dynamics -> residual)",
    y=0.98,
)
plt.tight_layout()
plt.show()

# Optional: save
# fig.savefig("spin_echo_bloch_schematic.png", dpi=200, bbox_inches="tight")

import numpy as np
import matplotlib.pyplot as plt


# ---------- Bloch-sphere helpers ----------
def Rx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_sphere(ax, n=18):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.5, alpha=0.25)


def draw_axes(ax, L=1.1):
    ax.plot([0, L], [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, L], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, 0], [0, L], linewidth=1)
    ax.text(L, 0, 0, "x", fontsize=9)
    ax.text(0, L, 0, "y", fontsize=9)
    ax.text(0, 0, L, "z", fontsize=9)


def draw_vec(ax, r, label=None):
    r = np.asarray(r, float)
    ax.quiver(0, 0, 0, r[0], r[1], r[2], arrow_length_ratio=0.12, linewidth=2)
    if label is not None:
        ax.text(r[0] * 1.05, r[1] * 1.05, r[2] * 1.05, label, fontsize=9)


def equator_arc(phi0, phi1, n=80):
    phis = np.linspace(phi0, phi1, n)
    return np.c_[np.cos(phis), np.sin(phis), np.zeros_like(phis)]


def azimuth(r):
    return np.arctan2(r[1], r[0])


# ---------- Spin-echo "cartoon" states ----------
# Set phi2 != phi1 to illustrate imperfect refocus (e.g., time-dependent noise / 13C dynamics).
phi1 = np.deg2rad(70)
phi2 = phi1  # try: phi2 = 0.8*phi1

r0 = np.array([0, 0, 1.0])  # init: ms=0
r1 = Rx(np.pi / 2) @ r0  # π/2x
r2 = Rz(phi1) @ r1  # free precession τ
r3 = Rx(np.pi) @ r2  # πx (refocus)
r4 = Rz(phi2) @ r3  # free precession τ
r5 = Rx(np.pi / 2) @ r4  # final π/2x (readout axis)

# ---------- Compact single-panel schematic ----------
fig = plt.figure(figsize=(5.2, 4.2))
ax = fig.add_subplot(1, 1, 1, projection="3d")

draw_sphere(ax, n=18)
draw_axes(ax)

# Vectors at key moments (numbered)
draw_vec(ax, r0, "0  init")
draw_vec(ax, r1, "1  π/2")
draw_vec(ax, r2, "2  τ")
draw_vec(ax, r3, "3  π")
draw_vec(ax, r4, "4  τ")
draw_vec(ax, r5, "5  π/2")

# Optional: show equator precession arcs (compact dashed curves)
arc1 = equator_arc(azimuth(r1), azimuth(r2))
arc2 = equator_arc(azimuth(r3), azimuth(r4))
ax.plot(arc1[:, 0], arc1[:, 1], arc1[:, 2], linestyle="--", linewidth=1)
ax.plot(arc2[:, 0], arc2[:, 1], arc2[:, 2], linestyle="--", linewidth=1)

ax.set_title("Spin Echo (Bloch sphere)", pad=6, fontsize=11)
ax.set_xlim([-1.15, 1.15])
ax.set_ylim([-1.15, 1.15])
ax.set_zlim([-1.15, 1.15])
ax.set_box_aspect([1, 1, 1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(elev=18, azim=35)

plt.tight_layout()
plt.show()

# fig.savefig("spin_echo_bloch_compact.png", dpi=250, bbox_inches="tight")
