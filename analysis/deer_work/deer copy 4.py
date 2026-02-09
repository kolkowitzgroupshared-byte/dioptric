#!/usr/bin/env python3
"""
DEER Bath Explorer: Interactive Parameter Study
================================================
Simulates NV-P1 DEER signal for your experimental setup with
tunable parameters:

  - P1 concentration
  - Evolution time (τ, number of revivals)
  - NV coherence time (T₂)
  - ESR contrast
  - P1 pulse duration (bandwidth)
  - Effective spin flip σ

Physics:
  - Collective bath: V(2τ) = C_Hahn × exp[-(2τ/T_DEER)^n]
  - Single P1 oscillation: cos(2π J σ τ_DEER)
  - NV echo envelope: exp(-2τ/T₂)
  - ¹³C revival condition: τ = k/(γ_13C × B)

Calibrated to Degen et al. Nat. Commun. 12, 3470 (2021):
  T_DEER = 0.77 ms at 75 ppb

Author: Generated for NV-P1 DEER experiment at ~52 G
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# DEFAULT PARAMETERS — CHANGE THESE FOR YOUR EXPERIMENT
# =============================================================================


class DEERParams:
    """All tunable experimental and physical parameters in one place."""

    def __init__(self):
        # ---- Magnetic field ----
        self.B_gauss = 52.15  # Total field magnitude [G]

        # ---- 13C revival ----
        self.gamma_13C_MHz_per_G = 1.0705e-3  # 13C gyromagnetic ratio [MHz/G]
        # Revival period = 1 / (gamma_13C * B)
        self.tau_revival_us = 1.0 / (self.gamma_13C_MHz_per_G * self.B_gauss)
        # ≈ 17.9 µs at 52.15 G

        # ---- NV properties ----
        self.T2_us = 100.0  # NV Hahn echo T₂ [µs]
        self.esr_contrast = 0.15  # Raw ESR contrast (fluorescence)
        self.pulse_fidelity = 0.80  # Combined fidelity of π/2, π, π/2 pulses
        self.pi_pulse_nv_ns = 128.0  # NV π pulse duration [ns]

        # ---- P1 RF pulse ----
        self.pi_pulse_p1_ns = 200.0  # P1 π pulse duration [ns]
        # Bandwidth (FWHM) = 0.89 / t_pulse
        self.p1_flip_probability = 1.0  # p = sin²(Ω t/2), 1.0 = perfect π pulse

        # ---- P1 bath properties ----
        self.sigma_eff = 0.15  # Effective spin flip |Δ⟨Sz⟩| at this field
        self.T_DEER_ref_ms = 0.77  # T_DEER at reference concentration [ms]
        self.conc_ref_ppb = 75.0  # Reference concentration [ppb]
        self.stretch_exponent = 2.0  # Stretched exponential (2 = Gaussian for 3D bath)

        # ---- Dipolar coupling constant ----
        # J(kHz) = dipolar_prefactor / r³(nm) × |1 - 3cos²α|
        self.dipolar_prefactor = 52.04  # kHz·nm³

        # ---- Diamond ----
        self.carbon_density = 1.764e23  # atoms/cm³

    @property
    def tau_revival_us_val(self):
        return 1.0 / (self.gamma_13C_MHz_per_G * self.B_gauss)

    @property
    def p1_bandwidth_MHz(self):
        return 0.89 / (self.pi_pulse_p1_ns / 1000.0)

    def hahn_contrast(self, two_tau_us):
        """NV Hahn echo contrast at given 2τ."""
        decay = np.exp(-two_tau_us / self.T2_us)
        return self.esr_contrast * decay * self.pulse_fidelity

    def T_DEER_ms(self, conc_ppb):
        """Bath dephasing time T_DEER [ms] at given P1 concentration."""
        return self.T_DEER_ref_ms * (self.conc_ref_ppb / conc_ppb)

    def bath_signal(self, tau_us, conc_ppb):
        """
        Collective bath DEER signal V(2τ) (normalized to 1).
        V = exp[-(2τ/T_DEER)^n]
        """
        T_ms = self.T_DEER_ms(conc_ppb)
        two_tau_ms = 2 * tau_us / 1000.0
        return np.exp(-((two_tau_ms / T_ms) ** self.stretch_exponent))

    def bath_dip_depth(self, tau_us, conc_ppb):
        """Fractional dip from collective bath."""
        return 1.0 - self.bath_signal(tau_us, conc_ppb)

    def single_p1_signal(self, tau_us, J_kHz):
        """
        Single-P1 DEER modulation.
        V = 1 - p × (1 - cos(2π J σ τ_DEER))
        where τ_DEER = τ (pump at center of echo).
        """
        tau_ms = tau_us / 1000.0
        phase = 2 * np.pi * J_kHz * self.sigma_eff * tau_ms  # dimensionless
        return 1.0 - self.p1_flip_probability * (1.0 - np.cos(phase))

    def single_p1_phase(self, tau_us, J_kHz):
        """Phase accumulated from single P1 [radians]."""
        tau_ms = tau_us / 1000.0
        return 2 * np.pi * J_kHz * self.sigma_eff * tau_ms

    def r_from_J(self, J_kHz, geo_factor=1.0):
        """NV-P1 distance [nm] from coupling J [kHz] and geometric factor."""
        return (self.dipolar_prefactor * geo_factor / J_kHz) ** (1.0 / 3.0)

    def J_from_r(self, r_nm, geo_factor=1.0):
        """Coupling J [kHz] from distance [nm] and geometric factor."""
        return self.dipolar_prefactor * geo_factor / (r_nm**3)

    def nn_distance(self, conc_ppb):
        """Mean nearest-neighbor distance [nm]."""
        n = conc_ppb * 1e-9 * self.carbon_density
        return 0.554 / (n ** (1.0 / 3.0)) * 1e7

    def min_detectable_phase(self, two_tau_us, num_averages=1e6, counts_per_shot=0.03):
        """
        Minimum detectable phase [rad] given experimental SNR.
        SNR on echo = C_Hahn × sqrt(N) × counts_per_shot_snr
        φ_min = sqrt(2/SNR_echo)
        """
        C = self.hahn_contrast(two_tau_us)
        snr_echo = C * np.sqrt(num_averages) * counts_per_shot
        if snr_echo <= 0:
            return np.inf
        return np.sqrt(2.0 / snr_echo)

    def print_summary(self):
        """Print current parameter summary."""
        print("=" * 70)
        print("DEER BATH EXPLORER — PARAMETER SUMMARY")
        print("=" * 70)
        print(f"  Magnetic field:        {self.B_gauss:.2f} G")
        print(f"  13C revival period:    {self.tau_revival_us_val:.2f} µs")
        print(f"  NV T₂:                {self.T2_us:.1f} µs")
        print(f"  ESR contrast:          {self.esr_contrast:.3f}")
        print(f"  Pulse fidelity:        {self.pulse_fidelity:.2f}")
        print(f"  NV π pulse:            {self.pi_pulse_nv_ns:.0f} ns")
        print(
            f"  P1 π pulse:            {self.pi_pulse_p1_ns:.0f} ns "
            f"(BW = {self.p1_bandwidth_MHz:.2f} MHz)"
        )
        print(f"  P1 flip probability:   {self.p1_flip_probability:.2f}")
        print(f"  Effective spin flip σ: {self.sigma_eff:.3f}")
        print(f"  T_DEER(75 ppb):        {self.T_DEER_ref_ms:.3f} ms")
        print(f"  Stretch exponent n:    {self.stretch_exponent:.1f}")
        print()


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================


def simulate_concentration_sweep(
    params, concentrations_ppb, revival_numbers, num_averages=1e6
):
    """
    Compute DEER dip depth vs concentration at different revival numbers.
    """
    tau_rev = params.tau_revival_us_val
    results = {}

    for k in revival_numbers:
        tau_us = k * tau_rev
        two_tau_us = 2 * tau_us
        C_hahn = params.hahn_contrast(two_tau_us)

        row = []
        for c in concentrations_ppb:
            bath_dip = params.bath_dip_depth(tau_us, c)
            measured_dip = C_hahn * bath_dip  # Absolute fluorescence change
            T_DEER = params.T_DEER_ms(c)
            phase_frac = (2 * tau_us / 1000.0) / T_DEER  # fraction of T_DEER

            row.append(
                {
                    "conc": c,
                    "k": k,
                    "tau_us": tau_us,
                    "two_tau_us": two_tau_us,
                    "C_hahn": C_hahn,
                    "bath_dip_frac": bath_dip,
                    "measured_dip": measured_dip,
                    "T_DEER_ms": T_DEER,
                    "phase_pi_frac": phase_frac,
                }
            )
        results[k] = row

    return results


def simulate_time_sweep(params, concentrations_ppb, tau_max_us=None):
    """
    Compute DEER signal vs evolution time for different concentrations.
    Shows both bath envelope and echo decay.
    """
    if tau_max_us is None:
        tau_max_us = 3 * params.T2_us

    tau_arr = np.linspace(0.1, tau_max_us, 2000)

    # Mark revival positions
    tau_rev = params.tau_revival_us_val
    max_revival = int(tau_max_us / tau_rev) + 1
    revival_taus = np.array([k * tau_rev for k in range(1, max_revival + 1)])

    results = {}
    for c in concentrations_ppb:
        V_bath = np.array([params.bath_signal(t, c) for t in tau_arr])
        echo_envelope = np.array([params.hahn_contrast(2 * t) for t in tau_arr])
        V_measured = echo_envelope * V_bath

        results[c] = {
            "tau": tau_arr,
            "V_bath": V_bath,
            "echo_envelope": echo_envelope,
            "V_measured": V_measured,
        }

    return results, revival_taus


def simulate_T2_sweep(params, T2_values_us, concentrations_ppb, revival_number=1):
    """
    Compute DEER sensitivity vs T₂ for different concentrations.
    Figure of merit: C_Hahn × bath_dip = measurable signal.
    """
    tau_rev = params.tau_revival_us_val
    tau_us = revival_number * tau_rev
    two_tau_us = 2 * tau_us

    results = {}
    for T2 in T2_values_us:
        # Temporarily override T2
        old_T2 = params.T2_us
        params.T2_us = T2
        C = params.hahn_contrast(two_tau_us)

        row = []
        for c in concentrations_ppb:
            dip = params.bath_dip_depth(tau_us, c)
            row.append(
                {
                    "T2": T2,
                    "conc": c,
                    "C_hahn": C,
                    "bath_dip": dip,
                    "measured_signal": C * dip,
                }
            )
        results[T2] = row
        params.T2_us = old_T2

    return results


def simulate_single_p1_vs_distance(params, distances_nm, revival_numbers):
    """
    Single-P1 DEER oscillation at different distances and revival numbers.
    """
    tau_rev = params.tau_revival_us_val
    results = {}

    for r in distances_nm:
        J = params.J_from_r(r, geo_factor=1.0)
        row = []
        for k in revival_numbers:
            tau_us = k * tau_rev
            phase = params.single_p1_phase(tau_us, J)
            V = params.single_p1_signal(tau_us, J)
            C = params.hahn_contrast(2 * tau_us)
            row.append(
                {
                    "r": r,
                    "k": k,
                    "tau_us": tau_us,
                    "J_kHz": J,
                    "phase_rad": phase,
                    "V_single": V,
                    "C_hahn": C,
                    "measured_dip": C * (1 - V),
                }
            )
        results[r] = row

    return results


# =============================================================================
# PLOTTING
# =============================================================================


def plot_all(params, save_dir="/home/claude"):
    """Generate all diagnostic plots."""

    concentrations = [5, 10, 25, 50, 75, 100, 200, 500, 1000]
    revival_numbers = [1, 2, 3, 4, 5]
    tau_rev = params.tau_revival_us_val

    cmap_conc = plt.cm.plasma
    cmap_rev = plt.cm.Set1

    # =================================================================
    # FIG 1: DEER signal vs time (bath + echo envelope)
    # =================================================================
    time_results, revival_taus = simulate_time_sweep(
        params, concentrations, tau_max_us=3 * params.T2_us
    )

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(6, 10), sharex=True)

    colors_c = [
        cmap_conc(i / (len(concentrations) - 1)) for i in range(len(concentrations))
    ]

    # Top: normalized bath signal
    for c, col in zip(concentrations, colors_c):
        ax1a.plot(
            time_results[c]["tau"],
            time_results[c]["V_bath"],
            color=col,
            lw=1.5,
            label=f"{c} ppb",
        )

    for rt in revival_taus[:6]:
        ax1a.axvline(rt, color="green", ls=":", lw=0.6, alpha=0.4)
    ax1a.axvline(tau_rev, color="green", ls=":", lw=1.2, alpha=0.7)
    ax1a.text(
        tau_rev + 0.5, 0.95, f"1st rev\n{tau_rev:.1f} µs", fontsize=8, color="green"
    )

    ax1a.axhline(1 / np.e, color="black", ls="--", lw=0.6, alpha=0.3)
    ax1a.set_ylabel("Bath signal V_bath(2τ)", fontsize=12)
    ax1a.set_title(
        f"DEER Bath Signal & Echo Envelope  |  B = {params.B_gauss:.1f} G,  "
        f"T₂ = {params.T2_us:.0f} µs,  σ = {params.sigma_eff:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    ax1a.legend(fontsize=8, ncol=3, loc="center right")
    ax1a.set_ylim(-0.02, 1.05)
    ax1a.grid(True, alpha=0.15)

    # Bottom: measured signal (bath × echo envelope)
    # Also show pure echo envelope
    tau_arr = time_results[concentrations[0]]["tau"]
    echo_env = time_results[concentrations[0]]["echo_envelope"]
    ax1b.plot(
        tau_arr, echo_env, "k--", lw=1.5, alpha=0.4, label="Echo envelope (no P1)"
    )

    for c, col in zip(concentrations, colors_c):
        ax1b.plot(
            time_results[c]["tau"],
            time_results[c]["V_measured"],
            color=col,
            lw=1.2,
            label=f"{c} ppb",
        )

    for rt in revival_taus[:6]:
        ax1b.axvline(rt, color="green", ls=":", lw=0.6, alpha=0.4)

    ax1b.set_xlabel("τ (µs)", fontsize=12)
    ax1b.set_ylabel("Measured signal (contrast × V_bath)", fontsize=12)
    ax1b.legend(fontsize=8, ncol=3, loc="upper right")
    ax1b.set_ylim(bottom=-0.005)
    ax1b.grid(True, alpha=0.15)

    plt.tight_layout()
    p1 = f"{save_dir}/bath_signal_vs_time.png"
    # fig1.savefig(p1, dpi=200, bbox_inches="tight")
    # plt.close(fig1)

    # =================================================================
    # FIG 2: Bath dip depth at each revival
    # =================================================================
    conc_results = simulate_concentration_sweep(params, concentrations, revival_numbers)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: fractional dip
    for ki, k in enumerate(revival_numbers):
        tau_us = k * tau_rev
        dips = [r["bath_dip_frac"] * 100 for r in conc_results[k]]
        col = cmap_rev(ki / max(len(revival_numbers) - 1, 1))
        ax2a.semilogy(
            concentrations,
            dips,
            "o-",
            color=col,
            lw=1.5,
            ms=5,
            label=f"k={k} (τ={tau_us:.1f} µs)",
        )

    ax2a.axhline(5, color="red", ls="--", lw=0.8, label="5% threshold")
    ax2a.axhline(1, color="orange", ls="--", lw=0.6, label="1% threshold")
    ax2a.set_xlabel("[P1] concentration (ppb)", fontsize=12)
    ax2a.set_ylabel("Bath dip depth (%)", fontsize=12)
    ax2a.set_title(
        "Fractional Bath Dip vs Concentration", fontsize=12, fontweight="bold"
    )
    ax2a.legend(fontsize=8)
    ax2a.grid(True, alpha=0.2, which="both")
    ax2a.set_xscale("log")

    # Right: measured dip (includes Hahn contrast)
    for ki, k in enumerate(revival_numbers):
        tau_us = k * tau_rev
        meas = [r["measured_dip"] * 100 for r in conc_results[k]]
        col = cmap_rev(ki / max(len(revival_numbers) - 1, 1))
        ax2b.semilogy(
            concentrations,
            meas,
            "o-",
            color=col,
            lw=1.5,
            ms=5,
            label=f'k={k} (C_Hahn={conc_results[k][0]["C_hahn"]:.3f})',
        )

    ax2b.set_xlabel("[P1] concentration (ppb)", fontsize=12)
    ax2b.set_ylabel("Measured dip (contrast × bath dip) (%)", fontsize=12)
    ax2b.set_title("Absolute Measurable DEER Dip", fontsize=12, fontweight="bold")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.2, which="both")
    ax2b.set_xscale("log")

    plt.tight_layout()
    p2 = f"{save_dir}/bath_dip_vs_concentration.png"
    # fig2.savefig(p2, dpi=200, bbox_inches="tight")
    # plt.close(fig2)

    # =================================================================
    # FIG 3: Sensitivity vs T₂
    # =================================================================
    T2_values = [30, 50, 75, 100, 150, 200, 300, 500, 1000]
    select_concs = [25, 75, 200, 500]

    fig3, axes3 = plt.subplots(
        1, len(revival_numbers[:3]), figsize=(16, 5), sharey=True
    )

    for ki, k in enumerate(revival_numbers[:3]):
        ax = axes3[ki]
        tau_us = k * tau_rev
        T2_results = simulate_T2_sweep(params, T2_values, select_concs, k)

        for ci, c in enumerate(select_concs):
            col = cmap_conc(ci / (len(select_concs) - 1))
            signals = [T2_results[t2][ci]["measured_signal"] * 100 for t2 in T2_values]
            ax.semilogy(
                T2_values, signals, "o-", color=col, lw=1.5, ms=4, label=f"{c} ppb"
            )

        ax.axhline(0.1, color="red", ls="--", lw=0.6, alpha=0.5)
        ax.axvline(params.T2_us, color="gray", ls=":", lw=1, alpha=0.5)
        ax.text(
            params.T2_us * 1.05,
            ax.get_ylim()[0] * 2,
            f"Your T₂",
            fontsize=8,
            color="gray",
        )

        ax.set_xlabel("NV T₂ (µs)", fontsize=11)
        if ki == 0:
            ax.set_ylabel("Measurable DEER signal (%)", fontsize=11)
        ax.set_title(
            f"Revival k={k} (τ={tau_us:.1f} µs)", fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, which="both")

    plt.suptitle(
        f"DEER Sensitivity vs NV T₂  |  ESR contrast = {params.esr_contrast}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    p3 = f"{save_dir}/sensitivity_vs_T2.png"
    # fig3.savefig(p3, dpi=200, bbox_inches="tight")
    # plt.close(fig3)

    # =================================================================
    # FIG 4: Single P1 phase accumulation at revivals
    # =================================================================
    distances = [5, 7, 10, 15, 20, 30]
    single_results = simulate_single_p1_vs_distance(params, distances, revival_numbers)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: phase vs revival number
    for di, r in enumerate(distances):
        col = cmap_conc(di / (len(distances) - 1))
        phases = [res["phase_rad"] for res in single_results[r]]
        ks = [res["k"] for res in single_results[r]]
        ax4a.plot(
            ks,
            phases,
            "o-",
            color=col,
            lw=1.5,
            ms=6,
            label=f"r={r} nm (J={params.J_from_r(r):.2f} kHz)",
        )

    ax4a.axhline(np.pi, color="red", ls="--", lw=1, label="π phase")
    ax4a.axhline(np.pi / 2, color="orange", ls="--", lw=0.8, label="π/2 phase")
    ax4a.set_xlabel("Revival number k", fontsize=12)
    ax4a.set_ylabel("Accumulated phase (rad)", fontsize=12)
    ax4a.set_title("Single P1: Phase vs Revival Number", fontsize=12, fontweight="bold")
    ax4a.legend(fontsize=8, ncol=2)
    ax4a.grid(True, alpha=0.2)

    # Right: measured dip vs revival
    for di, r in enumerate(distances):
        col = cmap_conc(di / (len(distances) - 1))
        meas_dips = [res["measured_dip"] * 100 for res in single_results[r]]
        ks = [res["k"] for res in single_results[r]]
        ax4b.plot(ks, meas_dips, "o-", color=col, lw=1.5, ms=6, label=f"r={r} nm")

    ax4b.set_xlabel("Revival number k", fontsize=12)
    ax4b.set_ylabel("Measured single-P1 dip (%)", fontsize=12)
    ax4b.set_title(
        "Single P1: Measurable Dip (includes echo decay)",
        fontsize=12,
        fontweight="bold",
    )
    ax4b.legend(fontsize=8)
    ax4b.grid(True, alpha=0.2)

    plt.tight_layout()
    p4 = f"{save_dir}/single_p1_phase.png"
    # fig4.savefig(p4, dpi=200, bbox_inches="tight")
    # plt.close(fig4)

    # =================================================================
    # FIG 5: Revival number optimization (figure of merit)
    # =================================================================
    fig5, ax5 = plt.subplots(figsize=(6, 5))

    ks_fine = np.arange(1, 15)
    select_concs_fom = [25, 50, 75, 100, 200, 500]

    for ci, c in enumerate(select_concs_fom):
        col = cmap_conc(ci / (len(select_concs_fom) - 1))
        foms = []
        for k in ks_fine:
            tau_us = k * tau_rev
            two_tau = 2 * tau_us
            C = params.hahn_contrast(two_tau)
            dip = params.bath_dip_depth(tau_us, c)
            # Figure of merit: C × dip (what you actually measure)
            fom = C * dip
            foms.append(fom * 100)  # percent
        ax5.plot(ks_fine, foms, "o-", color=col, lw=1.5, ms=5, label=f"{c} ppb")

    # Mark optimal revival for each
    for ci, c in enumerate(select_concs_fom):
        col = cmap_conc(ci / (len(select_concs_fom) - 1))
        foms = []
        for k in ks_fine:
            tau_us = k * tau_rev
            C = params.hahn_contrast(2 * tau_us)
            dip = params.bath_dip_depth(tau_us, c)
            foms.append(C * dip)
        best_k = ks_fine[np.argmax(foms)]
        best_fom = max(foms) * 100
        ax5.plot(best_k, best_fom, "*", color=col, ms=15, zorder=10)
        ax5.annotate(
            f"k={best_k}",
            (best_k, best_fom),
            textcoords="offset points",
            xytext=(8, 3),
            fontsize=8,
            color=col,
        )

    ax5.set_xlabel(f"Revival number k  (τ = k × {tau_rev:.1f} µs)", fontsize=12)
    ax5.set_ylabel("Measurable DEER signal (%)\n(C_Hahn × bath dip)", fontsize=12)
    ax5.set_title(
        f"Optimal Revival Number  |  T₂ = {params.T2_us:.0f} µs,  "
        f"ESR contrast = {params.esr_contrast}",
        fontsize=13,
        fontweight="bold",
    )
    ax5.legend(fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.2)
    ax5.set_xticks(ks_fine)

    plt.tight_layout()
    p5 = f"{save_dir}/optimal_revival.png"
    # fig5.savefig(p5, dpi=200, bbox_inches="tight")
    # plt.close(fig5)

    # =================================================================
    # FIG 6: Comprehensive heatmap — dip depth(concentration, revival)
    # =================================================================
    conc_fine = np.logspace(np.log10(5), np.log10(5000), 60)
    ks_heat = np.arange(1, 12)
    Z = np.zeros((len(ks_heat), len(conc_fine)))

    for ki, k in enumerate(ks_heat):
        for ci, c in enumerate(conc_fine):
            tau_us = k * tau_rev
            C = params.hahn_contrast(2 * tau_us)
            dip = params.bath_dip_depth(tau_us, c)
            Z[ki, ci] = C * dip * 100  # percent

    fig6, ax6 = plt.subplots(figsize=(6, 5))
    im = ax6.pcolormesh(conc_fine, ks_heat, Z, cmap="inferno", shading="auto")
    cb = plt.colorbar(im, ax=ax6, label="Measurable DEER signal (%)")

    # Contour lines
    CS = ax6.contour(
        conc_fine,
        ks_heat,
        Z,
        levels=[0.1, 0.5, 1, 2, 5],
        colors="white",
        linewidths=0.8,
        linestyles="--",
    )
    ax6.clabel(CS, fmt="%.1f%%", fontsize=8, colors="white")

    ax6.set_xlabel("[P1] concentration (ppb)", fontsize=12)
    ax6.set_ylabel(f"Revival number k  (τ = k × {tau_rev:.1f} µs)", fontsize=12)
    ax6.set_xscale("log")
    ax6.set_title(
        f"DEER Signal Map  |  T₂ = {params.T2_us:.0f} µs,  "
        f"σ = {params.sigma_eff},  ESR = {params.esr_contrast}",
        fontsize=13,
        fontweight="bold",
    )
    ax6.set_yticks(ks_heat)

    plt.tight_layout()
    p6 = f"{save_dir}/deer_heatmap.png"
    # fig6.savefig(p6, dpi=200, bbox_inches="tight")
    # plt.close(fig6)

    return [p1, p2, p3, p4, p5, p6]


# =============================================================================
# TEXT REPORT
# =============================================================================


def print_report(params):
    """Print comprehensive numerical report."""
    params.print_summary()

    tau_rev = params.tau_revival_us_val
    concentrations = [5, 10, 25, 50, 75, 100, 200, 500, 1000]
    revival_numbers = [1, 2, 3, 4, 5]

    # ---- Table 1: Bath dip at each revival ----
    print("=" * 90)
    print("TABLE 1: COLLECTIVE BATH DIP (%) at each revival")
    print("=" * 90)
    header = f"{'[P1] ppb':>10}"
    for k in revival_numbers:
        tau = k * tau_rev
        header += f"  {'k=%d (%.0fµs)' % (k, tau):>15}"
    print(header)
    print("-" * 90)

    for c in concentrations:
        row = f"{c:10d}"
        for k in revival_numbers:
            tau_us = k * tau_rev
            dip = params.bath_dip_depth(tau_us, c) * 100
            row += f"  {dip:15.4f}"
        print(row)

    # ---- Table 2: Measured signal (includes Hahn contrast) ----
    print()
    print("=" * 90)
    print("TABLE 2: MEASURED DEER SIGNAL (%) = C_Hahn × bath_dip")
    print("=" * 90)
    header = f"{'[P1] ppb':>10}"
    for k in revival_numbers:
        tau = k * tau_rev
        C = params.hahn_contrast(2 * tau)
        header += f"  {'k=%d (C=%.3f)' % (k, C):>15}"
    print(header)
    print("-" * 90)

    for c in concentrations:
        row = f"{c:10d}"
        for k in revival_numbers:
            tau_us = k * tau_rev
            C = params.hahn_contrast(2 * tau_us)
            dip = params.bath_dip_depth(tau_us, c)
            meas = C * dip * 100
            row += f"  {meas:15.5f}"
        print(row)

    # ---- Table 3: Phase accumulation from bath ----
    print()
    print("=" * 90)
    print("TABLE 3: PHASE ACCUMULATION (× π) from collective bath")
    print("=" * 90)
    header = f"{'[P1] ppb':>10}  {'T_DEER (µs)':>12}"
    for k in revival_numbers:
        header += f"  {'k=%d' % k:>8}"
    print(header)
    print("-" * 90)

    for c in concentrations:
        T = params.T_DEER_ms(c) * 1000  # µs
        row = f"{c:10d}  {T:12.1f}"
        for k in revival_numbers:
            tau_us = k * tau_rev
            phase_frac = (2 * tau_us) / T
            row += f"  {phase_frac:8.4f}"
        print(row)

    # ---- Table 4: Single P1 at various distances ----
    print()
    print("=" * 90)
    print("TABLE 4: SINGLE P1 — phase (rad) at 1st revival (τ = %.1f µs)" % tau_rev)
    print("=" * 90)
    distances = [3, 5, 7, 10, 12, 15, 20, 25, 30]
    print(
        f"{'r (nm)':>8}  {'J (kHz)':>10}  {'phase (rad)':>12}  {'phase/π':>8}  "
        f"{'dip (frac)':>10}  {'× C_Hahn':>10}  {'Need k= for π':>14}"
    )
    print("-" * 90)

    C1 = params.hahn_contrast(2 * tau_rev)
    for r in distances:
        J = params.J_from_r(r)
        phase = params.single_p1_phase(tau_rev, J)
        V = params.single_p1_signal(tau_rev, J)
        dip = 1 - V
        meas = C1 * dip
        # Revival number needed for π phase
        if phase > 0:
            k_pi = np.pi / phase
        else:
            k_pi = np.inf
        k_pi_str = f"{k_pi:.1f}" if k_pi < 50 else ">50"

        print(
            f"{r:8.1f}  {J:10.4f}  {phase:12.6f}  {phase/np.pi:8.4f}  "
            f"{dip:10.6f}  {meas:10.6f}  {k_pi_str:>14}"
        )

    # ---- Optimal revival ----
    print()
    print("=" * 90)
    print("TABLE 5: OPTIMAL REVIVAL NUMBER (maximizes C_Hahn × bath_dip)")
    print("=" * 90)
    print(
        f"{'[P1] ppb':>10}  {'Best k':>7}  {'τ_opt (µs)':>11}  "
        f"{'C_Hahn':>8}  {'Bath dip':>10}  {'Signal (%)':>11}"
    )
    print("-" * 70)

    for c in concentrations:
        best_k = 1
        best_signal = 0
        for k in range(1, 20):
            tau_us = k * tau_rev
            C = params.hahn_contrast(2 * tau_us)
            dip = params.bath_dip_depth(tau_us, c)
            sig = C * dip
            if sig > best_signal:
                best_signal = sig
                best_k = k
        tau_opt = best_k * tau_rev
        C_opt = params.hahn_contrast(2 * tau_opt)
        dip_opt = params.bath_dip_depth(tau_opt, c)
        print(
            f"{c:10d}  {best_k:7d}  {tau_opt:11.1f}  "
            f"{C_opt:8.4f}  {dip_opt:10.5f}  {best_signal*100:11.5f}"
        )

    print()


# =============================================================================
# MAIN
# =============================================================================


def main():
    # ============================================================
    # CREATE PARAMETER OBJECT — MODIFY HERE FOR YOUR EXPERIMENT
    # ============================================================
    p = DEERParams()

    # --- Override any defaults here ---
    p.B_gauss = 52.15  # Your field [G]
    p.T2_us = 100.0  # Your NV T₂ [µs]
    p.esr_contrast = 0.15  # Your ESR contrast
    p.pulse_fidelity = 0.80  # Estimated pulse fidelity
    p.pi_pulse_nv_ns = 128.0  # Your NV π pulse [ns]
    p.pi_pulse_p1_ns = 200.0  # Your P1 RF pulse [ns]
    p.sigma_eff = 0.15  # Effective spin flip at 52 G
    p.T2_us = 100.0  # NV T₂ [µs]

    # ============================================================
    # RUN
    # ============================================================
    save_dir = "/home/claude"

    print_report(p)

    print("Generating plots...")
    paths = plot_all(p, save_dir=save_dir)
    for path in paths:
        print(f"  Saved: {path}")

    print("\nDone!")
    return p


if __name__ == "__main__":
    p = main()
    plt.show()
