#!/usr/bin/env python3
"""
Phase accumulation vs time at different P1 concentrations.
Shows the collective bath phase and marks revival positions.
"""
import numpy as np
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parameters
B_gauss = 52.15
gamma_13C = 1.0705e-3  # MHz/G
tau_rev_us = 1.0 / (gamma_13C * B_gauss)  # ~17.9 µs
T2_us = 100.0
T_DEER_ref_ms = 0.77
conc_ref = 75.0

concentrations = [10, 25, 50, 75, 100, 200, 500, 1000]


def T_DEER(c):
    return T_DEER_ref_ms * (conc_ref / c)  # ms


def bath_phase_pi(tau_us, c):
    """Effective phase in units of π: 2τ / T_DEER."""
    return (2 * tau_us / 1000.0) / T_DEER(c)


def echo_envelope(tau_us):
    return np.exp(-2 * tau_us / T2_us)


# Time axis — continuous
tau = np.linspace(0.1, 300, 3000)  # µs

# Revival positions
revivals = np.arange(1, 17) * tau_rev_us

cmap = plt.cm.plasma
colors = [cmap(i / (len(concentrations) - 1)) for i in range(len(concentrations))]

# =================================================================
# FIGURE: Phase (in units of π) vs τ
# =================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# --- Top: continuous phase vs time ---
for c, col in zip(concentrations, colors):
    phase = bath_phase_pi(tau, c)
    ax1.plot(
        tau, phase, color=col, lw=1.8, label=f"{c} ppb (T_DEER={T_DEER(c)*1000:.0f} µs)"
    )

# Mark π thresholds
for n, ls in [(0.25, ":"), (0.5, "--"), (1.0, "-"), (2.0, "-")]:
    ax1.axhline(n, color="gray", ls=ls, lw=0.7, alpha=0.4)
    ax1.text(tau[-1] * 1.01, n, f"{n}π", fontsize=9, va="center", color="gray")

# Mark revivals
for k, rt in enumerate(revivals, 1):
    ax1.axvline(rt, color="green", ls=":", lw=0.5, alpha=0.3)
    if k <= 8:
        ax1.text(
            rt,
            ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 3.5,
            f"k={k}",
            fontsize=7,
            ha="center",
            va="bottom",
            color="green",
            alpha=0.6,
        )

# Mark T2
ax1.axvline(T2_us / 2, color="red", ls="--", lw=1, alpha=0.5)
ax1.text(T2_us / 2 + 1, 0.1, "T₂/2", fontsize=9, color="red", alpha=0.7)

ax1.set_ylabel("Accumulated phase (× π)", fontsize=12)
ax1.set_title(
    f"Collective Bath Phase Accumulation vs Evolution Time\n"
    f"B = {B_gauss:.1f} G,  τ_revival = {tau_rev_us:.1f} µs,  T₂ = {T2_us:.0f} µs",
    fontsize=13,
    fontweight="bold",
)
ax1.legend(fontsize=8, ncol=2, loc="upper left")
ax1.set_ylim(0, 3.5)
ax1.grid(True, alpha=0.15)

# --- Bottom: phase at revival points only (what you can actually measure) ---
for c, col in zip(concentrations, colors):
    phases_at_rev = [bath_phase_pi(rt, c) for rt in revivals]
    ax2.plot(revivals, phases_at_rev, "o-", color=col, lw=1.2, ms=5, label=f"{c} ppb")

    # Also show echo-weighted phase (phase × echo amplitude)
    weighted = [bath_phase_pi(rt, c) * echo_envelope(rt) for rt in revivals]
    ax2.plot(revivals, weighted, "s--", color=col, lw=0.8, ms=3, alpha=0.4)

for n, ls in [(0.25, ":"), (0.5, "--"), (1.0, "-")]:
    ax2.axhline(n, color="gray", ls=ls, lw=0.7, alpha=0.4)
    ax2.text(revivals[-1] + 2, n, f"{n}π", fontsize=9, va="center", color="gray")

ax2.axvline(T2_us / 2, color="red", ls="--", lw=1, alpha=0.5)

# Add second y-axis showing revival number
ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
ax2_top.set_xticks(revivals[:12])
ax2_top.set_xticklabels([f"k={k}" for k in range(1, 13)], fontsize=7, rotation=45)

ax2.set_xlabel("τ (µs)", fontsize=12)
ax2.set_ylabel("Phase at revival (× π)", fontsize=12)
ax2.set_title(
    "Phase at Revival Points  (solid = raw,  dashed = echo-weighted)", fontsize=11
)
ax2.legend(fontsize=8, ncol=2, loc="upper left")
# ax2.set_ylim(0, 3.0)
ax2.grid(True, alpha=0.15)

plt.tight_layout()
# fig.savefig("/home/claude/phase_vs_time.png", dpi=200, bbox_inches="tight")
# plt.close(fig)
print("Saved: phase_vs_time.png")

# =================================================================
# FIGURE 2: Time to reach specific phase milestones
# =================================================================
fig2, ax = plt.subplots(figsize=(6, 5))

conc_sweep = np.logspace(np.log10(5), np.log10(5000), 200)

milestones = [
    (0.1, "π/10 (detectable)", "green", "-"),
    (0.25, "π/4", "blue", "-"),
    (0.5, "π/2", "orange", "-"),
    (1.0, "π", "red", "-"),
    (2.0, "2π", "darkred", "--"),
]

for phase_pi, label, color, ls in milestones:
    # τ where 2τ/T_DEER = phase_pi → τ = phase_pi × T_DEER / 2
    tau_needed = [phase_pi * T_DEER(c) * 1000 / 2 for c in conc_sweep]  # µs
    ax.loglog(conc_sweep, tau_needed, color=color, ls=ls, lw=1.8, label=label)

# Mark T2/2 and revival positions
ax.axhline(T2_us / 2, color="gray", ls=":", lw=1.2)
ax.text(5.5, T2_us / 2 * 1.1, f"T₂/2 = {T2_us/2:.0f} µs", fontsize=9, color="gray")

for k in [1, 2, 3, 5]:
    rt = k * tau_rev_us
    ax.axhline(rt, color="green", ls=":", lw=0.6, alpha=0.4)
    ax.text(
        4500, rt * 1.05, f"k={k} ({rt:.0f} µs)", fontsize=8, color="green", ha="right"
    )

# Shade feasible region (τ < T₂/2)
ax.fill_between(conc_sweep, 0.1, T2_us / 2, alpha=0.06, color="green")
ax.text(20, 5, "Measurable\n(τ < T₂/2)", fontsize=10, color="green", alpha=0.5)

ax.set_xlabel("[P1] concentration (ppb)", fontsize=12)
ax.set_ylabel("τ needed (µs)", fontsize=12)
ax.set_title(
    f"Time to Reach Phase Milestone vs P1 Concentration\n"
    f"T₂ = {T2_us:.0f} µs,  revival = {tau_rev_us:.1f} µs",
    fontsize=13,
    fontweight="bold",
)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(5, 5000)
ax.set_ylim(0.5, 5000)
ax.grid(True, alpha=0.2, which="both")

plt.tight_layout()
# fig2.savefig("/home/claude/time_to_phase.png", dpi=200, bbox_inches="tight")
# plt.close(fig2)
print("Saved: time_to_phase.png")

# =================================================================
# Print summary table
# =================================================================
print()
print("=" * 100)
print("PHASE AT EACH REVIVAL (× π)")
print("=" * 100)
header = f"{'[P1]':>8}"
for k in range(1, 9):
    header += f"  {'k=%d (%.0fµs)' % (k, k*tau_rev_us):>14}"
print(header)
print("-" * 100)

for c in concentrations:
    row = f"{c:>7}p"
    for k in range(1, 9):
        ph = bath_phase_pi(k * tau_rev_us, c)
        marker = " *" if ph > 1.0 else ("  " if ph < 0.1 else "")
        row += f"  {ph:12.4f}{marker}"
    print(row)

print()
print("  * = exceeds π (full dephasing)")
print(f"  Revival period = {tau_rev_us:.2f} µs")
print(f"  T₂/2 = {T2_us/2:.0f} µs → max useful revival k ≈ {int(T2_us/2/tau_rev_us)}")

plt.show()
