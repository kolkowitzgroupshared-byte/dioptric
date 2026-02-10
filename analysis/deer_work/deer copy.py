#!/usr/bin/env python3
"""
DEER Time-Domain Simulation: Collective Bath Dephasing
======================================================
Simulates the NV spin-echo signal V(τ) under DEER for different P1
concentrations, showing how the collective bath produces detectable
dips and the timescale to accumulate phase.

Physics: V(2τ) = exp[-(2τ/T_DEER)^n] * cos-modulation from discrete spins
  - T_DEER scales as 1/[P1] (from 3D dipolar bath theory)
  - Calibrated to Degen et al.: T_DEER ≈ 0.77 ms at 75 ppb
  - Stretched exponential with n ~ 2 (Gaussian-like for 3D bath)
"""

import numpy as np

# import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =============================================================================
# CONSTANTS
# =============================================================================
CARBON_DENSITY = 1.764e23  # cm⁻³
T_DEER_REF = 0.77  # ms, at 75 ppb (from Degen et al. Fig 1c)
CONC_REF = 75.0  # ppb reference
STRETCH_EXP = 2.0  # Gaussian-like decay for 3D bath

# =============================================================================
# DEER TIME-DOMAIN MODEL
# =============================================================================


def t_deer(conc_ppb):
    """T_DEER in ms, scaling linearly with 1/concentration."""
    return T_DEER_REF * (CONC_REF / conc_ppb)


def nearest_neighbor_dist(conc_ppb):
    """Mean nearest-neighbor distance in nm."""
    n = conc_ppb * 1e-9 * CARBON_DENSITY
    return 0.554 / (n ** (1.0 / 3.0)) * 1e7


def j_coupling_nn(conc_ppb, sigma_eff=0.15):
    """Nearest-neighbor NV-P1 coupling in kHz (angular avg)."""
    r_nm = nearest_neighbor_dist(conc_ppb)
    return 52.04 * 0.8 / (r_nm**3)


def deer_signal_bath(tau_ms, conc_ppb, n_exponent=STRETCH_EXP):
    """
    Collective bath DEER signal V(2τ).
    Stretched exponential: V = exp(-(2τ/T_DEER)^n)
    """
    T = t_deer(conc_ppb)
    return np.exp(-((2 * tau_ms / T) ** n_exponent))


def deer_signal_with_oscillation(tau_ms, conc_ppb, J_kHz=None, sigma=0.15):
    """
    DEER signal including both bath envelope and nearest-neighbor oscillation.
    V(2τ) = exp(-(2τ/T_DEER)^n) * [1 - p*(1-cos(2π J σ 2τ))]
    where p accounts for the probability the nearest P1 is in the right state (~1/12).
    """
    envelope = deer_signal_bath(tau_ms, conc_ppb)

    if J_kHz is None:
        J_kHz = j_coupling_nn(conc_ppb, sigma)

    # Nearest neighbor oscillation (probability ~1/12 to be in right state)
    p_nn = 1.0 / 12.0
    phase = 2 * np.pi * J_kHz * 1e-3 * sigma * 2 * tau_ms  # dimensionless
    oscillation = 1 - p_nn * (1 - np.cos(phase))

    return envelope * oscillation


def tau_for_dip(conc_ppb, dip_fraction=0.05):
    """Time τ (ms) to reach a given fractional dip depth."""
    # 1 - exp(-(2τ/T)^n) = dip_fraction
    # (2τ/T)^n = -ln(1-dip_fraction)
    T = t_deer(conc_ppb)
    return 0.5 * T * (-np.log(1 - dip_fraction)) ** (1.0 / STRETCH_EXP)


def tau_for_pi_phase_bath(conc_ppb):
    """Time τ (ms) where bath dephasing reaches ~π (signal → 0, V ≈ e⁻¹)."""
    return 0.5 * t_deer(conc_ppb)  # at 2τ = T_DEER, V = e⁻¹ ≈ 0.37


# =============================================================================
# MAIN
# =============================================================================

concentrations = [5, 10, 25, 50, 75, 100, 200, 500, 1000]
tau_max_ms = 5.0
tau = np.linspace(0, tau_max_ms, 2000)

# ---- Print summary table ----
print("=" * 95)
print("DEER TIME-DOMAIN: COLLECTIVE BATH DEPHASING")
print("=" * 95)
print(
    f"  Reference: T_DEER = {T_DEER_REF} ms at {CONC_REF:.0f} ppb (Degen et al. 2021)"
)
print(f"  Decay: stretched exponential with n = {STRETCH_EXP}")
print()
print(
    f"{'[P1] ppb':>10}  {'r_nn (nm)':>10}  {'J_nn (kHz)':>11}  {'T_DEER (ms)':>12}  "
    f"{'τ(5% dip)':>10}  {'τ(20% dip)':>11}  {'τ(1/e dip)':>11}  {'NV T2~1ms?'}"
)
print(f"{'─'*10}  {'─'*10}  {'─'*11}  {'─'*12}  {'─'*10}  {'─'*11}  {'─'*11}  {'─'*12}")

for c in concentrations:
    r = nearest_neighbor_dist(c)
    J = j_coupling_nn(c)
    T = t_deer(c)
    t5 = tau_for_dip(c, 0.05)
    t20 = tau_for_dip(c, 0.20)
    t1e = tau_for_pi_phase_bath(c)

    feasible = "Yes" if t5 < 1.0 else ("Borderline" if t5 < 2.0 else "No")

    print(
        f"{c:10d}  {r:10.1f}  {J:11.4f}  {T:12.3f}  "
        f"{t5:10.3f}  {t20:11.3f}  {t1e:11.3f}  {feasible}"
    )

print()
print("  τ(5% dip): 2τ where signal drops by 5%")
print("  τ(20% dip): 2τ where signal drops by 20%")
print("  τ(1/e dip): 2τ where signal = 1/e ≈ 0.37 (effective π dephasing)")
print("  NV T2 ~ 1 ms assumed (Degen et al.)")

# =============================================================================
# FIGURE 1: Bath envelope V(2τ) for all concentrations
# =============================================================================
fig1, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

cmap = plt.cm.plasma
colors = [cmap(i / (len(concentrations) - 1)) for i in range(len(concentrations))]

for c, col in zip(concentrations, colors):
    V = deer_signal_bath(tau, c)
    ax1.plot(tau, V, color=col, lw=1.8, label=f"{c} ppb")

# Mark T2 limit
ax1.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax1.text(
    0.52,
    0.95,
    "NV T₂/2\n≈ 0.5 ms",
    fontsize=9,
    color="gray",
    transform=ax1.get_xaxis_transform(),
    va="top",
)

ax1.axhline(1 / np.e, color="black", ls="--", lw=0.8, alpha=0.4)
ax1.text(tau_max_ms * 0.98, 1 / np.e + 0.02, "1/e", fontsize=9, ha="right", alpha=0.5)

ax1.axhline(0.95, color="green", ls="--", lw=0.6, alpha=0.4)
ax1.text(
    tau_max_ms * 0.98, 0.96, "5% dip", fontsize=8, ha="right", color="green", alpha=0.6
)

ax1.set_ylabel("DEER signal V(2τ)", fontsize=12)
ax1.set_ylim(-0.02, 1.05)
ax1.legend(fontsize=9, ncol=3, loc="center right")
ax1.set_title(
    "Collective P1 Bath: DEER Time-Domain Signal\n"
    f"V(2τ) = exp[−(2τ/T_DEER)²],  T_DEER = {T_DEER_REF} ms × (75 ppb / [P1])",
    fontsize=13,
    fontweight="bold",
)
ax1.grid(True, alpha=0.2)

# Bottom panel: dip depth = 1 - V
for c, col in zip(concentrations, colors):
    V = deer_signal_bath(tau, c)
    ax2.plot(tau, (1 - V) * 100, color=col, lw=1.5)

ax2.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax2.axhline(5, color="green", ls="--", lw=0.6, alpha=0.4)
ax2.text(
    tau_max_ms * 0.98,
    5.5,
    "5% detection threshold",
    fontsize=8,
    ha="right",
    color="green",
    alpha=0.6,
)

ax2.set_xlabel("τ (ms)", fontsize=12)
ax2.set_ylabel("Dip depth (%)", fontsize=12)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
# fig1.savefig("/home/claude/deer_time_domain_bath.png", dpi=200, bbox_inches="tight")
# plt.close(fig1)
print("\nSaved: deer_time_domain_bath.png")

# =============================================================================
# FIGURE 2: Including nearest-neighbor oscillation
# =============================================================================
fig2, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for idx, (c, col) in enumerate(zip(concentrations, colors)):
    ax = axes[idx]

    V_bath = deer_signal_bath(tau, c)
    V_full = deer_signal_with_oscillation(tau, c)

    ax.plot(
        tau, V_bath, color="gray", lw=1.0, ls="--", alpha=0.6, label="Bath envelope"
    )
    ax.plot(tau, V_full, color=col, lw=1.5, label="Bath + nearest P1")

    T = t_deer(c)
    J = j_coupling_nn(c)
    r = nearest_neighbor_dist(c)

    ax.set_title(f"[P1] = {c} ppb", fontsize=11, fontweight="bold", color=col)
    info = f"T_DEER = {T:.2f} ms\nJ_nn = {J:.4f} kHz\nr_nn = {r:.1f} nm"
    ax.text(
        0.97,
        0.95,
        info,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )

    ax.axvline(0.5, color="gray", ls=":", lw=0.6, alpha=0.4)
    ax.set_ylim(max(0, min(V_full) - 0.05), 1.05)
    ax.grid(True, alpha=0.15)

    if idx >= 6:
        ax.set_xlabel("τ (ms)", fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel("V(2τ)", fontsize=10)
    if idx == 0:
        ax.legend(fontsize=8, loc="lower left")

fig2.suptitle(
    "DEER Signal: Bath Envelope + Nearest-Neighbor Oscillation",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
# fig2.savefig("/home/claude/deer_time_domain_panels.png", dpi=200, bbox_inches="tight")
# plt.close(fig2)
print("Saved: deer_time_domain_panels.png")

# =============================================================================
# FIGURE 3: T_DEER and detection threshold vs concentration
# =============================================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

conc_sweep = np.logspace(np.log10(1), np.log10(10000), 200)

# Left: T_DEER vs concentration
T_sweep = [t_deer(c) for c in conc_sweep]
ax3a.loglog(conc_sweep, T_sweep, "b-", lw=2)
ax3a.axhline(1.0, color="red", ls="--", lw=1, label="NV T₂ ≈ 1 ms")
ax3a.axhline(0.5, color="orange", ls="--", lw=1, label="NV T₂/2 ≈ 0.5 ms")
for c in [5, 75, 200, 1000]:
    T = t_deer(c)
    ax3a.plot(c, T, "ko", ms=6)
    ax3a.annotate(
        f"{c} ppb\n({T:.2f} ms)",
        (c, T),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=8,
    )
ax3a.set_xlabel("[P1] concentration (ppb)", fontsize=12)
ax3a.set_ylabel("T_DEER (ms)", fontsize=12)
ax3a.set_title("Bath Dephasing Time vs Concentration", fontsize=13, fontweight="bold")
ax3a.legend(fontsize=10)
ax3a.grid(True, alpha=0.2, which="both")
ax3a.set_xlim(1, 10000)

# Right: τ needed for different dip depths
for dip, color, ls in [
    (0.01, "green", "-"),
    (0.05, "blue", "-"),
    (0.20, "orange", "-"),
    (0.50, "red", "-"),
]:
    tau_sweep = [tau_for_dip(c, dip) for c in conc_sweep]
    ax3b.loglog(
        conc_sweep, tau_sweep, color=color, ls=ls, lw=1.8, label=f"{dip*100:.0f}% dip"
    )

ax3b.axhline(0.5, color="gray", ls=":", lw=1)
ax3b.text(1.2, 0.55, "τ = T₂/2", fontsize=9, color="gray")

# Shade feasible region
ax3b.fill_between(conc_sweep, 0.001, 0.5, alpha=0.08, color="green")
ax3b.text(
    300,
    0.01,
    "Detectable\n(τ < T₂/2)",
    fontsize=10,
    color="green",
    ha="center",
    alpha=0.6,
)

ax3b.set_xlabel("[P1] concentration (ppb)", fontsize=12)
ax3b.set_ylabel("Required τ (ms)", fontsize=12)
ax3b.set_title(
    "Interaction Time for Detectable DEER Dip", fontsize=13, fontweight="bold"
)
ax3b.legend(fontsize=10)
ax3b.grid(True, alpha=0.2, which="both")
ax3b.set_xlim(1, 10000)
ax3b.set_ylim(0.001, 100)

plt.tight_layout()
# fig3.savefig("/home/claude/deer_detection_threshold.png", dpi=200, bbox_inches="tight")
# plt.close(fig3)
print("Saved: deer_detection_threshold.png")

# =============================================================================
# FIGURE 4: Experimental comparison — what you'd measure at τ = 0.3 ms
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(12, 5))

tau_fixed = 0.3  # ms — typical experimental value
bar_concs = [5, 10, 25, 50, 75, 100, 200, 500, 1000]
dips = [(1 - deer_signal_bath(tau_fixed, c)) * 100 for c in bar_concs]
bar_colors = [cmap(i / (len(bar_concs) - 1)) for i in range(len(bar_concs))]

bars = ax4.bar(
    [str(c) for c in bar_concs], dips, color=bar_colors, edgecolor="black", lw=0.5
)
ax4.axhline(5, color="red", ls="--", lw=1, label="5% detection threshold")
ax4.axhline(1, color="orange", ls="--", lw=0.8, label="1% (challenging)")

for bar, d in zip(bars, dips):
    if d > 1:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            d + 0.5,
            f"{d:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    else:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            d + 0.3,
            f"{d:.2f}%",
            ha="center",
            fontsize=8,
            color="gray",
        )

ax4.set_xlabel("[P1] concentration (ppb)", fontsize=12)
ax4.set_ylabel("DEER dip depth (%)", fontsize=12)
ax4.set_title(
    f"Expected DEER Dip Depth at τ = {tau_fixed} ms (fixed interaction time)",
    fontsize=13,
    fontweight="bold",
)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.15, axis="y")

plt.tight_layout()
# fig4.savefig("/home/claude/deer_dip_vs_concentration.png", dpi=200, bbox_inches="tight")
# plt.close(fig4)
print("Saved: deer_dip_vs_concentration.png")

print("\nDone!")
plt.show()
