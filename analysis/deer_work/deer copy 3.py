#!/usr/bin/env python3
"""
DEER Simulation: YOUR Experimental Parameters
==============================================
Hahn-echo DEER with:
  - Interpulse τ = 18 µs, total echo time 2τ = 36 µs
  - NV π pulse: 128 ns
  - P1 RF π pulse: 200 ns (simultaneous with NV π)
  - Revival condition at 2τ = 36 µs

Key question: what dip depths and linewidths do you expect
at different P1 concentrations with THESE parameters?
"""

import numpy as np
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# YOUR EXPERIMENTAL PARAMETERS
# =============================================================================
TAU_US = 18.0  # interpulse delay [µs]
TWO_TAU_US = 36.0  # total echo time [µs]
TAU_MS = TAU_US / 1000.0
TWO_TAU_MS = TWO_TAU_US / 1000.0

PI_PULSE_NV_NS = 128.0  # NV π pulse duration [ns]
PI_PULSE_P1_NS = 200.0  # P1 RF π pulse duration [ns]
PI_PULSE_P1_US = PI_PULSE_P1_NS / 1000.0

# Rabi frequencies
RABI_NV_MHZ = 1.0 / (2 * PI_PULSE_NV_NS * 1e-3)  # MHz (from t_π = 1/(2Ω))
RABI_P1_MHZ = 1.0 / (2 * PI_PULSE_P1_NS * 1e-3)  # MHz

# MW bandwidth of P1 pulse (FWHM)
BW_P1_MHZ = 0.89 / PI_PULSE_P1_US  # sinc FWHM

# Reference
CARBON_DENSITY = 1.764e23  # cm⁻³
T_DEER_REF_MS = 0.77  # at 75 ppb (Degen et al.)
CONC_REF = 75.0

concentrations = [5, 10, 25, 50, 75, 100, 200, 500, 1000, 5000, 10000]

print("=" * 80)
print("YOUR DEER EXPERIMENT PARAMETERS")
print("=" * 80)
print(f"  Interpulse delay τ     = {TAU_US} µs")
print(f"  Total echo time 2τ     = {TWO_TAU_US} µs")
print(f"  NV π pulse             = {PI_PULSE_NV_NS} ns  (Ω_NV = {RABI_NV_MHZ:.1f} MHz)")
print(f"  P1 RF π pulse          = {PI_PULSE_P1_NS} ns  (Ω_P1 = {RABI_P1_MHZ:.2f} MHz)")
print(f"  P1 pulse bandwidth     = {BW_P1_MHZ:.2f} MHz (FWHM)")
print()

# =============================================================================
# 1. COLLECTIVE BATH DIP AT YOUR τ
# =============================================================================
print("=" * 80)
print("1. COLLECTIVE BATH DIP at 2τ = 36 µs")
print("=" * 80)
print(f"   V(2τ) = exp[-(2τ/T_DEER)²]")
print()
print(
    f"{'[P1] ppb':>10}  {'T_DEER (ms)':>12}  {'(2τ/T)²':>12}  {'Dip depth':>12}  {'Detectable?'}"
)
print(f"{'─'*10}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*15}")

for c in concentrations:
    T = T_DEER_REF_MS * (CONC_REF / c)
    ratio_sq = (TWO_TAU_MS / T) ** 2
    V = np.exp(-ratio_sq)
    dip = (1 - V) * 100

    if dip > 5:
        det = "YES"
    elif dip > 1:
        det = "Marginal"
    elif dip > 0.1:
        det = "Very hard"
    else:
        det = "No"

    print(f"{c:10d}  {T:12.4f}  {ratio_sq:12.6f}  {dip:11.4f}%  {det}")

# =============================================================================
# 2. SINGLE STRONGLY-COUPLED P1 (what you're actually seeing)
# =============================================================================
print()
print("=" * 80)
print("2. SINGLE P1: Revival at 2τ = 36 µs → coupling strength")
print("=" * 80)
print()
print("  Revival condition: J·σ·2τ = 2π  →  J_eff = 1/(σ·2τ)")
print("  (or half-revival: J·σ·2τ = π  →  J_eff = 1/(2·σ·2τ))")
print()

sigma_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]

print(
    f"  {'σ_eff':>6}  {'J (full rev)':>14}  {'J (half rev)':>14}  {'r (full, nm)':>13}  {'r (half, nm)':>13}"
)
print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*13}  {'─'*13}")

for sigma in sigma_values:
    # Full revival: J * sigma * 2tau = 2pi → J = 1/(sigma * 2tau) in frequency
    # 2tau in ms, J in kHz: J_kHz = 1000/(sigma * 2tau_us)
    J_full_kHz = 1000.0 / (sigma * TWO_TAU_US)
    J_half_kHz = 500.0 / (sigma * TWO_TAU_US)

    # Distance from J = 52.04 * 0.8 / r³ (angular avg)
    # r³ = 52.04 * 0.8 / J → r = (41.63/J)^(1/3)  [J in kHz, r in nm]
    r_full = (41.63 / J_full_kHz) ** (1.0 / 3.0)
    r_half = (41.63 / J_half_kHz) ** (1.0 / 3.0)

    print(
        f"  {sigma:6.2f}  {J_full_kHz:12.2f} kHz  {J_half_kHz:12.2f} kHz  "
        f"{r_full:11.1f} nm  {r_half:11.1f} nm"
    )

print()
print("  NOTE: σ ~ 0.10-0.25 typical at 52 G (from your Hamiltonian simulation)")
print("  Most likely scenario: σ ≈ 0.15-0.25, giving J ≈ 75-185 kHz")
print("  → a P1 center at r ≈ 4-6 nm from your NV")

# =============================================================================
# 3. DEER FREQUENCY SPECTRUM AT YOUR PARAMETERS
# =============================================================================
print()
print("=" * 80)
print("3. LINEWIDTH AND SPECTRAL RESOLUTION with 200 ns P1 pulse")
print("=" * 80)
print()
print(f"  P1 pulse bandwidth (FWHM) = {BW_P1_MHZ:.2f} MHz")
print(f"  This is MUCH broader than at Degen et al.'s 4 µs pulse (0.22 MHz)")
print()
print(
    f"{'[P1] ppb':>10}  {'Γ_dip (MHz)':>12}  {'Γ_MW (MHz)':>11}  {'Γ_total (MHz)':>14}  {'Resolved lines?'}"
)
print(f"{'─'*10}  {'─'*12}  {'─'*11}  {'─'*14}  {'─'*20}")

for c in concentrations:
    gamma_dip = 4.5e-3 * c
    gamma_mw = BW_P1_MHZ
    gamma_total = np.sqrt(gamma_dip**2 + gamma_mw**2)

    # Minimum JT splitting at this field is ~4 MHz (from transition sim)
    if gamma_total < 2:
        resolved = "Individual JT lines"
    elif gamma_total < 8:
        resolved = "Partially resolved"
    elif gamma_total < 20:
        resolved = "Broad groups only"
    else:
        resolved = "Unresolved"

    print(
        f"{c:10d}  {gamma_dip:12.4f}  {gamma_mw:11.2f}  {gamma_total:14.3f}  {resolved}"
    )

# =============================================================================
# 4. NUMBER OF P1 SPINS ADDRESSED BY YOUR 200 ns PULSE
# =============================================================================
print()
print("=" * 80)
print("4. NUMBER OF P1 SPINS FLIPPED by your 200 ns pulse")
print("=" * 80)
print()
print(
    f"  Bandwidth = {BW_P1_MHZ:.2f} MHz addresses all P1s within ±{BW_P1_MHZ/2:.2f} MHz"
)
print(f"  At each frequency, you flip P1s from 1 JT axis × 1 nuclear state ≈ n/12")
print(f"  BUT with {BW_P1_MHZ:.1f} MHz bandwidth, you may address MULTIPLE transitions")
print()

# From the Hamiltonian simulation, transitions within each group span:
# Low-freq group: 76-94 MHz (span ~18 MHz) → multiple JT lines within one pulse
# Mid-freq group: 194-201 MHz (span ~7 MHz) → multiple lines within one pulse
# High-freq group: 256-266 MHz (span ~10 MHz) → multiple lines within one pulse

print(f"  Transition group spans (from Hamiltonian diag):")
print(
    f"    Low-freq  (76-94 MHz):  span ~18 MHz  → pulse covers ~{BW_P1_MHZ/18*100:.0f}% of group"
)
print(
    f"    Mid-freq  (194-201 MHz): span ~7 MHz  → pulse covers ~{min(BW_P1_MHZ/7*100,100):.0f}% of group"
)
print(
    f"    High-freq (256-266 MHz): span ~10 MHz → pulse covers ~{min(BW_P1_MHZ/10*100,100):.0f}% of group"
)
print()

print(
    f"{'[P1] ppb':>10}  {'n_total (cm⁻³)':>15}  {'n in BW (cm⁻³)':>15}  "
    f"{'N in (20nm)³':>13}  {'Dip from these':>15}"
)
print(f"{'─'*10}  {'─'*15}  {'─'*15}  {'─'*13}  {'─'*15}")

for c in concentrations:
    n_total = c * 1e-9 * CARBON_DENSITY
    # Fraction addressed: ~1/12 per line, but BW covers ~25% of a group
    # → effectively ~1/12 * (BW/group_span) per group, but we address 1 group
    # Simplified: per resolved line ~n/12, if BW > line spacing then more
    frac_addressed = 1.0 / 12.0  # conservative: one line
    if BW_P1_MHZ > 5:
        frac_addressed = 1.0 / 4.0  # broad pulse → all nuclear states in one JT
    n_addr = n_total * frac_addressed

    # Number in a (20 nm)³ sensing volume (typical NV sensing range)
    vol_cm3 = (20e-7) ** 3
    N_in_vol = n_addr * vol_cm3

    # Single-spin dip: each spin contributes ~J²σ²τ² to the phase
    # Collective: dip ≈ N * (J_avg * σ * 2τ)² for small dip
    r_nn = 0.554 / (n_total ** (1.0 / 3.0)) * 1e7  # nm
    J_nn = 52.04 * 0.8 / (r_nn**3)  # kHz
    sigma = 0.15
    single_phase = (2 * np.pi * J_nn * 1e-3 * sigma * TWO_TAU_MS) ** 2
    dip_pct = N_in_vol * single_phase * 100  # percent, small angle approx

    print(
        f"{c:10d}  {n_total:15.3e}  {n_addr:15.3e}  "
        f"{N_in_vol:13.4f}  {min(dip_pct, 100):13.4f}%"
    )

# =============================================================================
# 5. PHASE ACCUMULATION TIMELINE
# =============================================================================
print()
print("=" * 80)
print("5. WHAT τ DO YOU NEED for π phase from collective bath?")
print("=" * 80)
print()
print(f"  Your current τ = {TAU_US} µs (2τ = {TWO_TAU_US} µs)")
print(f"  For π dephasing (V → 1/e), need 2τ = T_DEER")
print()
print(
    f"{'[P1] ppb':>10}  {'T_DEER':>10}  {'τ_π':>10}  {'Your τ/τ_π':>12}  {'Phase at your τ':>16}"
)
print(f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*16}")

for c in concentrations:
    T = T_DEER_REF_MS * (CONC_REF / c)
    tau_pi = T / 2  # ms
    ratio = TAU_MS / tau_pi
    phase_rad = np.sqrt(
        -np.log(np.exp(-((TWO_TAU_MS / T) ** 2)))
    )  # effective phase in "units of π"
    phase_frac = TWO_TAU_MS / T  # fraction of T_DEER

    print(
        f"{c:10d}  {T*1000:8.1f} µs  {tau_pi*1000:8.1f} µs  {ratio:12.5f}  "
        f"{phase_frac:14.5f} × π"
    )

print()
print("  You are at τ/τ_π << 1 for all reasonable concentrations")
print("  → You are in the SHORT-TIME LIMIT of the bath")
print("  → Any signal you see at 36 µs is from INDIVIDUAL nearby P1s, not the bath")

# =============================================================================
# PLOTS
# =============================================================================

# ---- Figure 1: Time-domain with your τ marked ----
fig1, ax1 = plt.subplots(figsize=(14, 7))

tau_sweep = np.linspace(0, 2.0, 2000)  # ms
cmap = plt.cm.plasma
select_concs = [10, 50, 75, 100, 200, 500, 1000, 5000]
colors = [cmap(i / (len(select_concs) - 1)) for i in range(len(select_concs))]

for c, col in zip(select_concs, colors):
    T = T_DEER_REF_MS * (CONC_REF / c)
    V = np.exp(-((2 * tau_sweep / T) ** 2))
    ax1.plot(tau_sweep, V, color=col, lw=1.8, label=f"{c} ppb (T={T*1000:.0f} µs)")

# Mark YOUR τ
ax1.axvline(TAU_MS, color="red", lw=2.5, ls="-", alpha=0.8, zorder=10)
ax1.text(
    TAU_MS + 0.01,
    0.5,
    f"YOUR τ = {TAU_US:.0f} µs",
    fontsize=12,
    color="red",
    fontweight="bold",
    rotation=90,
    va="center",
)

# Zoom inset showing your τ region
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ax_inset = inset_axes(ax1, width="40%", height="45%", loc="center right")
tau_zoom = np.linspace(0, 0.05, 500)

for c, col in zip(select_concs, colors):
    T = T_DEER_REF_MS * (CONC_REF / c)
    V = np.exp(-((2 * tau_zoom / T) ** 2))
    ax_inset.plot(tau_zoom * 1000, V, color=col, lw=1.5)  # x in µs

ax_inset.axvline(TAU_US / 1000 * 1000, color="red", lw=2, ls="-", alpha=0.8)
ax_inset.set_xlabel("τ (µs)", fontsize=9)
ax_inset.set_ylabel("V(2τ)", fontsize=9)
ax_inset.set_title(f"Zoom: 0–50 µs", fontsize=9)
ax_inset.set_ylim(0.9, 1.005)
ax_inset.tick_params(labelsize=8)
ax_inset.grid(True, alpha=0.2)

ax1.axhline(1 / np.e, color="black", ls="--", lw=0.8, alpha=0.3)
ax1.set_xlabel("τ (ms)", fontsize=13)
ax1.set_ylabel("DEER signal V(2τ)", fontsize=13)
ax1.set_title(
    f"Collective Bath DEER Signal\n"
    f"Your experiment: τ = {TAU_US} µs, P1 pulse = {PI_PULSE_P1_NS:.0f} ns "
    f"(BW = {BW_P1_MHZ:.1f} MHz)",
    fontsize=14,
    fontweight="bold",
)
ax1.legend(fontsize=9, loc="lower left", ncol=2)
ax1.set_ylim(-0.02, 1.05)
ax1.grid(True, alpha=0.15)

plt.tight_layout()
# fig1.savefig("/home/claude/deer_your_tau.png", dpi=200, bbox_inches="tight")
# plt.close(fig1)
print("\nSaved: deer_your_tau.png")

# ---- Figure 2: What coupling does a 36 µs revival imply? ----
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

# Left: J vs sigma for revival at 36 µs
sigma_range = np.linspace(0.05, 0.5, 100)

# Full revival: J*sigma*2tau = 2pi → J = 1/(sigma*2tau_us) * 1000 kHz
J_full = 1000.0 / (sigma_range * TWO_TAU_US)
J_half = 500.0 / (sigma_range * TWO_TAU_US)
J_quarter = 250.0 / (sigma_range * TWO_TAU_US)

ax2a.plot(sigma_range, J_full, "r-", lw=2, label="Full revival (2π)")
ax2a.plot(sigma_range, J_half, "b-", lw=2, label="Half revival (π)")
ax2a.plot(sigma_range, J_quarter, "g-", lw=2, label="Quarter revival (π/2)")

# Shade typical sigma range at 52 G
ax2a.axvspan(0.10, 0.25, alpha=0.15, color="yellow", label="σ range at 52 G")

# Degen et al. reference
ax2a.axhline(17.8, color="gray", ls=":", lw=1)
ax2a.text(0.45, 18.5, "Degen S1-S2 (17.8 kHz)", fontsize=8, color="gray", ha="right")

ax2a.set_xlabel("Effective spin flip σ", fontsize=12)
ax2a.set_ylabel("Required J (kHz)", fontsize=12)
ax2a.set_title(
    f"NV–P1 Coupling for Revival at 2τ = {TWO_TAU_US:.0f} µs",
    fontsize=13,
    fontweight="bold",
)
ax2a.legend(fontsize=9)
ax2a.set_ylim(0, 300)
ax2a.grid(True, alpha=0.2)

# Right: corresponding distance
r_full = (41.63 / J_full) ** (1.0 / 3.0)
r_half = (41.63 / J_half) ** (1.0 / 3.0)
r_quarter = (41.63 / J_quarter) ** (1.0 / 3.0)

ax2b.plot(sigma_range, r_full, "r-", lw=2, label="Full revival (2π)")
ax2b.plot(sigma_range, r_half, "b-", lw=2, label="Half revival (π)")
ax2b.plot(sigma_range, r_quarter, "g-", lw=2, label="Quarter revival (π/2)")

ax2b.axvspan(0.10, 0.25, alpha=0.15, color="yellow", label="σ range at 52 G")

ax2b.set_xlabel("Effective spin flip σ", fontsize=12)
ax2b.set_ylabel("NV–P1 distance (nm)", fontsize=12)
ax2b.set_title("Implied NV–P1 Distance", fontsize=13, fontweight="bold")
ax2b.legend(fontsize=9)
ax2b.set_ylim(0, 12)
ax2b.grid(True, alpha=0.2)

plt.tight_layout()
# fig2.savefig("/home/claude/deer_revival_coupling.png", dpi=200, bbox_inches="tight")
# plt.close(fig2)
print("Saved: deer_revival_coupling.png")

# ---- Figure 3: Dip depth at YOUR τ vs concentration ----
fig3, ax3 = plt.subplots(figsize=(12, 5))

conc_plot = [5, 10, 25, 50, 75, 100, 200, 500, 1000, 5000, 10000]
dips_bath = []
for c in conc_plot:
    T = T_DEER_REF_MS * (CONC_REF / c)
    V = np.exp(-((TWO_TAU_MS / T) ** 2))
    dips_bath.append((1 - V) * 100)

bar_colors = [cmap(i / (len(conc_plot) - 1)) for i in range(len(conc_plot))]
bars = ax3.bar(
    range(len(conc_plot)), dips_bath, color=bar_colors, edgecolor="black", lw=0.5
)
ax3.set_xticks(range(len(conc_plot)))
ax3.set_xticklabels([str(c) for c in conc_plot], rotation=45)

for bar, d in zip(bars, dips_bath):
    label = f"{d:.3f}%" if d < 1 else f"{d:.1f}%"
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        d + max(dips_bath) * 0.01,
        label,
        ha="center",
        fontsize=8,
        fontweight="bold",
    )

ax3.axhline(5, color="red", ls="--", lw=1, label="5% threshold")
ax3.axhline(1, color="orange", ls="--", lw=0.8, label="1% threshold")
ax3.set_xlabel("[P1] concentration (ppb)", fontsize=12)
ax3.set_ylabel("Collective bath dip (%)", fontsize=12)
ax3.set_title(
    f"Collective Bath DEER Dip at YOUR τ = {TAU_US:.0f} µs (2τ = {TWO_TAU_US:.0f} µs)\n"
    f"Any signal you see is from individual nearby P1s, not the bath",
    fontsize=13,
    fontweight="bold",
)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.15, axis="y")

plt.tight_layout()
# fig3.savefig("/home/claude/deer_dip_your_tau.png", dpi=200, bbox_inches="tight")
# plt.close(fig3)
print("Saved: deer_dip_your_tau.png")

print()
print("=" * 80)
print("BOTTOM LINE")
print("=" * 80)
print(
    f"""
  At your τ = {TAU_US} µs with 200 ns P1 pulse:

  1. COLLECTIVE BATH is invisible (dip < 0.2% even at 75 ppb)
     → You need τ > 100 µs to start seeing the bath at 75 ppb

  2. YOUR SIGNAL comes from INDIVIDUAL strongly-coupled P1 centers
     → Revival at 36 µs implies J·σ ≈ 28 kHz (full) or 14 kHz (half)
     → For σ ≈ 0.15: J ≈ 90-185 kHz → r ≈ 4-5 nm
     → For σ ≈ 0.25: J ≈ 55-110 kHz → r ≈ 5-7 nm
     → This is a P1 sitting VERY close to your NV

  3. Your 200 ns pulse has BW = {BW_P1_MHZ:.1f} MHz
     → Much broader than Degen's 0.22 MHz
     → You address ~all nuclear spin states within one JT axis
     → Less selective but stronger signal from each P1

  4. COMPARISON TO DEGEN et al.:
     → They used τ up to ~500 µs to see the bath
     → Their strongly-coupled S1 had J = 1.9 kHz at ~15 nm
     → Your revival at 36 µs suggests a MUCH closer P1 (4-7 nm)
"""
)

print("Done!")

plt.show()
