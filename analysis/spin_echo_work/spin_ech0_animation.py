import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# Single-spin Spin-Echo animation (transparent background, faint Bloch sphere)
# π/2 – τ – π – τ
#
# NOTE: A single spin doesn't "dephase" by itself; dephasing is an ensemble/noise effect.
# For talk visuals, we show "dephasing" as shrinking transverse coherence during first τ,
# and "refocusing" as regrowth during second τ after the π pulse.
# -----------------------------

# Parameters (tweak these)
pi2_frames = 18
tau_frames = 60
pi_frames = 8
fps = 25
dephase_depth = 0.75  # 0..1: how much the transverse component shrinks in free1
detuning = 0.12  # rad/frame (sets rotation speed on equator)

# Build time segments: pi/2 -> free(tau) -> pi -> free(tau)
segments = [
    ("pi2", pi2_frames),
    ("free1", tau_frames),
    ("pi", pi_frames),
    ("free2", tau_frames),
]
T = sum(fr for _, fr in segments)

frame_kind, frame_prog = [], []
for kind, fr in segments:
    for k in range(fr):
        frame_kind.append(kind)
        frame_prog.append((k + 1) / fr)

phi = 0.0  # phase in x-y plane


def rotate_y(v, angle):
    """Rotate 3D vector v around y-axis by angle."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = v
    return np.array([c * x + s * z, y, -s * x + c * z])


def make_bloch_sphere(ax, alpha=0.18, lw=0.6):
    """Draw a faint Bloch sphere wireframe."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    wf = ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=lw)
    wf.set_alpha(alpha)


def hide_3d_axes(ax):
    """Remove axes/panes/grid for a clean transparent animation."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_alpha(0.0)
        axis._axinfo["grid"]["linewidth"] = 0


def set_line(ln, vec):
    """Set a 3D line from origin to vec."""
    ln.set_data([0, vec[0]], [0, vec[1]])
    ln.set_3d_properties([0, vec[2]])


# Figure setup (transparent)
fig = plt.figure(figsize=(6, 6))
fig.patch.set_alpha(0.0)  # transparent figure background
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor((0, 0, 0, 0))  # transparent axes background
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_zlim(-1.05, 1.05)
hide_3d_axes(ax)
ax.view_init(elev=18, azim=35)

make_bloch_sphere(ax, alpha=0.18, lw=0.6)

# Spin vector + tip marker
(line,) = ax.plot([0, 0], [0, 0], [0, 1], linewidth=3)
tip = ax.scatter([0], [0], [1], s=40)


def update(frame):
    global phi
    kind = frame_kind[frame]
    s = frame_prog[frame]

    # π/2 pulse: rotate from +z to +x about y-axis
    if kind == "pi2":
        v = rotate_y(np.array([0.0, 0.0, 1.0]), (np.pi / 2) * s)
        set_line(line, v)
        tip._offsets3d = ([v[0]], [v[1]], [v[2]])
        return line, tip

    # Free evolution: phase accumulation on equator
    if kind == "free1":
        phi += detuning
        r = 1.0 - dephase_depth * s  # shrink transverse component (visual "dephasing")
    elif kind == "pi":
        # Apply π at end of pulse: invert phase accumulation (toggling frame)
        if s >= 1.0 - 1e-9:
            phi = -phi
        r = 1.0 - dephase_depth
    else:  # free2
        phi += detuning
        r = 1.0 - dephase_depth * (1.0 - s)  # regrow (visual "refocus")

    x = float(r * np.cos(phi))
    y = float(r * np.sin(phi))
    z = 0.0

    v = np.array([x, y, z])
    set_line(line, v)
    tip._offsets3d = ([x], [y], [z])
    return line, tip


anim = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

# Save as transparent GIF
anim.save(
    "spin_echo_bloch_3d_transparent_single_spin.gif",
    writer=PillowWriter(fps=fps),
    savefig_kwargs={"transparent": True, "facecolor": "none"},
)

print("Saved: spin_echo_bloch_3d_transparent_single_spin.gif")
