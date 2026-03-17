# -*- coding: utf-8 -*-
"""
Spin-echo simulator for a single NV with flexible 13C configuration.

Key features:
- Fixed NV frame & B-field (no accidental re-randomization of orientation)
- Choose exact 13C sites (fixed_site_ids) or a full mask (fixed_presence_mask)
- Or Bernoulli-thin once and reuse (quenched) / redraw each time (ensemble)
- Correct scalar A_parallel and B_perp; angular frequency in Mk_tau
- Optional phenomenological fine-decay comb (can disable)

Dependencies: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import data_manager as dm
from utils import kplotlib as kpl
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Dict, Optional
import json
import shutil
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter, PillowWriter

# ---------- Optional numba (falls back gracefully) ----------
try:
    from numba import njit
except Exception:

    def njit(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap


# ---------- Physical constants ----------
D_NV = 2.87e9  # Hz (unused here, but kept for completeness)
gamma_e = 28e9  # Hz/T (unused here)
gamma_C13 = 10.705e6  # Hz/T  (13C gyromagnetic ratio)

# Default lab field in Gauss (example), rotated later into NV frame
# B_vec_G = np.array([-46.18287122, -17.44411563, -5.57779074], dtype=float) ##49.68G
# B_vec_G = np.array([-31.61263115, -56.58135644, -6.5512002 ], dtype=float) ##65.14G
# B_vec_G = np.array([-41.57848995, -32.77145194, -27.5799348 ], dtype=float) ##59.69G
B_vec_G = np.array([-48.67047318, -32.07615947, 22.49657427], dtype=float)  ##62.48G
B_vec_T = B_vec_G * 1e-4


# =============================================================================
# Fine-decay (phenomenological)
# =============================================================================
def fine_decay(
    tau_us,
    baseline=1.0,
    comb_contrast=0.6,
    revival_time=37.0,
    width0_us=6.0,
    T2_ms=0.08,
    T2_exp=1.0,
    amp_taper_alpha=0.0,
    width_slope=0.0,
    revival_chirp=0.0,
    # NEW (additive, signed; zero-mean cos carrier(s))
    osc_amp=0.0,
    osc_f0=0.0,
    osc_phi0=0.0,
    osc_f1=0.0,
    osc_phi1=0.0,
):
    """
    signal(τ) = baseline
                - comb_contrast * envelope(τ) * COMB(τ)               [dip]
                +                 envelope(τ) * COMB(τ) * OSC(τ)      [signed oscillation]

    envelope(τ) = exp[-(τ / (1000*T2_ms))^T2_exp]
    COMB(τ)     = Σ_k [ 1/(1+k)^amp_taper_alpha ] * exp(-((τ-μ_k)/w_k)^4)
                    μ_k = k * revival_time * (1 + k*revival_chirp)
                    w_k = width0_us * (1 + k*width_slope)

    OSC(τ)      = osc_amp * [ cos(2π f0 τ + φ0) + cos(2π f1 τ + φ1) ]   # zero mean
    τ in microseconds, f in cycles/μs, phases in rad.
    """
    tau = np.asarray(tau_us, dtype=float).ravel()
    width0_us = max(1e-9, float(width0_us))
    revival_time = max(1e-9, float(revival_time))
    T2_us = max(1e-9, 1000.0 * float(T2_ms))
    T2_exp = float(T2_exp)

    # envelope
    envelope = np.exp(-((tau / T2_us) ** T2_exp))

    # number of revivals to include
    tau_max = float(np.nanmax(tau)) if tau.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / revival_time)) + 1))

    comb = _comb_quartic_powerlaw(
        tau,
        revival_time,
        width0_us,
        amp_taper_alpha,
        width_slope,
        revival_chirp,
        n_guess,
    )

    # beating lives in MOD; comb_contrast is the overall amplitude (once)
    if (osc_amp != 0.0) and (osc_f0 != 0.0 or osc_f1 != 0.0):
        s0 = np.sin(np.pi * osc_f0 * tau + osc_phi0)
        s1 = np.sin(np.pi * osc_f1 * tau + osc_phi1)
        beat = (s0 * s0) * (s1 * s1)
        mod = comb_contrast - osc_amp * beat
    else:
        mod = comb_contrast

    return baseline - envelope * mod * comb


@njit
def _comb_quartic_powerlaw(
    tau, revival_time, width0_us, amp_taper_alpha, width_slope, revival_chirp, n_guess
):
    n = tau.shape[0]
    out = np.zeros(n, dtype=np.float64)
    tmax = 0.0
    for i in range(n):
        if tau[i] > tmax:
            tmax = tau[i]

    for k in range(n_guess):
        mu_k = k * revival_time * (1.0 + k * revival_chirp)
        w_k = width0_us * (1.0 + k * width_slope)
        if w_k <= 0.0:
            continue
        if mu_k > tmax + 5.0 * w_k:
            break

        amp_k = 1.0 / ((1.0 + k) ** amp_taper_alpha)
        inv_w4 = 1.0 / (w_k**4)

        for i in range(n):
            x = tau[i] - mu_k
            out[i] += amp_k * np.exp(-(x * x) * (x * x) * inv_w4)

    return out


# =============================================================================
# NOISE MODELING
# =============================================================================
@dataclass
class NoiseSpec:
    # photon/shot model (approximate binomial with variance ~ p(1-p)/N)
    use_shot: bool = True
    photons_mean: float = 3_000.0  # mean detected photons per point at y=1

    # additive electronics noise
    add_gauss_std: float = 0.003  # ~0.3% RMS

    # slow drifts per trace
    mult_gain_std: float = 0.02  # multiplicative gain ~ N(1,σ)
    base_offset_std: float = 0.01  # additive offset ~ N(0,σ)

    # correlated noise across tau
    use_1f: bool = True
    onef_strength: float = 0.004  # overall RMS of 1/f component
    onef_alpha: float = 1.0  # spectral slope: 0=white, 1=pink

    use_ar1: bool = False  # alternative to 1/f if you prefer
    ar1_rho: float = 0.98  # correlation
    ar1_sigma: float = 0.002  # innovations std

    # timing jitter (distorts x-axis slightly)
    timing_jitter_std_us: float = 0.01  # std of τ jitter (μs), applied pointwise

    # small per-trace frequency detuning (mimics MW/B-field drift)
    freq_detune_ppm: float = 200.0  # ppm of MHz-scale features; set 0 to disable

    # rare spikes/outliers
    glitch_prob: float = 0.001
    glitch_magnitude: float = 0.1  # absolute kick

    # quantization
    quantize_bits: int = 0  # 0 to disable; else 12, 14, 16, ...

    # bounds
    clip_lo: float = -0.1  # allow slight under/overshoot pre-clip
    clip_hi: float = 1.2


def _gen_1f_noise(T: int, rng: np.random.Generator, alpha=1.0, rms=0.005):
    """Synthesize length-T 1/f^alpha noise with target RMS."""
    # frequency grid
    freqs = np.fft.rfftfreq(T)
    amp = np.ones_like(freqs)
    # avoid division by zero at DC; leave DC random but small
    amp[1:] = 1.0 / (freqs[1:] ** (alpha / 2.0))  # /2 because we’ll mirror energy
    # complex spectrum with random phases
    phases = rng.uniform(0, 2 * np.pi, size=freqs.shape)
    spec = amp * (np.cos(phases) + 1j * np.sin(phases))
    # inverse FFT to time domain
    x = np.fft.irfft(spec, n=T)
    # normalize to unit RMS then scale
    x = x / (np.std(x) + 1e-12)
    return rms * x


def _gen_ar1_noise(T: int, rng: np.random.Generator, rho=0.98, sigma=0.002):
    e = rng.standard_normal(T) * sigma
    x = np.empty(T, float)
    x[0] = e[0] / max(1e-6, (1 - rho))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


def apply_noise_pipeline(
    y_clean: np.ndarray, taus_us: np.ndarray, rng: np.random.Generator, spec: NoiseSpec
) -> np.ndarray:
    y = y_clean.astype(float, copy=True)
    T = y.shape[0]

    # (0) small per-trace detuning → stretch tau slightly
    if spec.freq_detune_ppm and np.any(np.diff(taus_us) > 0):
        scale = 1.0 + 1e-6 * spec.freq_detune_ppm * rng.standard_normal()
        # resample y at tau' = tau * scale
        tau_scaled = taus_us * scale
        y = np.interp(taus_us, tau_scaled, y, left=y[0], right=y[-1])

    # (1) multiplicative gain & baseline offset (per trace)
    if spec.mult_gain_std > 0:
        gain = 1.0 + spec.mult_gain_std * rng.standard_normal()
        y *= gain
    if spec.base_offset_std > 0:
        y += spec.base_offset_std * rng.standard_normal()

    # (2) correlated noise across τ
    if spec.use_1f and spec.onef_strength > 0:
        y += _gen_1f_noise(T, rng, alpha=spec.onef_alpha, rms=spec.onef_strength)
    elif spec.use_ar1:
        y += _gen_ar1_noise(T, rng, rho=spec.ar1_rho, sigma=spec.ar1_sigma)

    # (3) timing jitter: perturb τ and resample (pointwise)
    if spec.timing_jitter_std_us > 0:
        dt = rng.standard_normal(T) * spec.timing_jitter_std_us
        tau_jit = np.clip(taus_us + dt, taus_us.min(), taus_us.max())
        tau_jit.sort()  # preserve monotonicity; yields slight local stretching
        y = np.interp(taus_us, tau_jit, y)

    # (4) shot/readout noise (approximate binomial/Poisson)
    if spec.use_shot and spec.photons_mean > 0:
        # approximate: counts ~ Poisson(mu = photons_mean * y_clipped)
        lam = np.clip(y, 0.0, 1.0) * spec.photons_mean
        counts = rng.poisson(lam)
        y = counts / max(1.0, spec.photons_mean)

    # (5) additive Gaussian electronics noise
    if spec.add_gauss_std > 0:
        y += spec.add_gauss_std * rng.standard_normal(T)

    # (6) rare glitches/outliers
    if spec.glitch_prob > 0 and spec.glitch_magnitude > 0:
        mask = rng.random(T) < spec.glitch_prob
        y[mask] += spec.glitch_magnitude * rng.standard_normal(mask.sum())

    # (7) quantization
    if spec.quantize_bits and spec.quantize_bits > 0:
        levels = 2**spec.quantize_bits
        lo, hi = 0.0, 1.0  # assume normalized signal; adjust if needed
        step = (hi - lo) / (levels - 1)
        y = np.round(np.clip(y, lo, hi) / step) * step

    # final clip
    return np.clip(y, spec.clip_lo, spec.clip_hi)


# =============================================================================
# Spin echo mapping at revivals
# =============================================================================
def _synthesize_comb_only(taus_sec, p):
    return fine_decay(
        tau_us=taus_sec * 1e6,
        baseline=float(p.get("baseline", 1.0)),
        comb_contrast=float(p.get("comb_contrast", 0.6)),
        revival_time=float(p.get("revival_time", 37.0)),
        width0_us=float(p.get("width0_us", 6.0)),
        T2_ms=float(p.get("T2_ms", 0.08)),
        T2_exp=float(p.get("T2_exp", 1.0)),
        amp_taper_alpha=float(p.get("amp_taper_alpha", 0.0)),
        width_slope=float(p.get("width_slope", 0.0)),
        revival_chirp=float(p.get("revival_chirp", 0.0)),
        osc_amp=0.0,  # <- explicitly no oscillation when synthesizing comb-only
        osc_f0=0.0,
        osc_f1=0.0,
        osc_phi0=0.0,
        osc_phi1=0.0,
    )


def add_charge_pedestal(
    y_core,
    taus_us,
    gate_G,
    *,
    A_ch=0.03,  # amplitude of the pedestal (0..~0.1 reasonable)
    T_ch_us=None,  # optional readout-specific decay; None→use no extra decay
    baseline=0.6,
):
    """
    y_core: baseline - depth(τ)*E(τ)   (your existing physical echo)
    gate_G: 0 at τ≈0, peaks at k*Trev  (same comb you already compute, normalized 0..1)
    A_ch  : pedestal amplitude (fraction of total scale)
    T_ch_us: if set, pedestal has its own envelope exp[-(τ/T_ch)^1] (often > T2_us)
    """
    tau = np.asarray(taus_us, float)
    G = np.clip(np.asarray(gate_G, float), 0.0, 1.0)

    # Optional slow decay for charge gain (often longer than spin T2)
    if T_ch_us is not None and T_ch_us > 0:
        E_ch = np.exp(-(tau / float(T_ch_us)))
    else:
        E_ch = 1.0

    P = A_ch * G * E_ch  # strictly >= 0

    # Headroom safety: don’t exceed physical maximum (≈1.0), but do it smoothly
    headroom = 1.0 - y_core
    P = np.minimum(P, np.maximum(0.0, headroom))

    return y_core + P


def apply_readout_gain(y_core, gate_G, *, beta=0.08, baseline=0.6):
    """
    y_core is already on your 0..1-ish PL scale with baseline ~0.6.
    We remap around baseline so that a gain >1 lifts toward/above baseline.
    """
    G = np.clip(np.asarray(gate_G, float), 0.0, 1.0)

    # deviation from baseline, then apply a gain that also adds a small offset piece:
    #   y_raw = baseline + (y_core - baseline) * (1 + beta*G) + beta*G*(1 - baseline)
    # The last term acts like a gain-induced offset in counts.
    y_out = (
        baseline + (y_core - baseline) * (1.0 + beta * G) + beta * G * (1.0 - baseline)
    )

    # Smooth headroom cap (no hard clip)
    return np.minimum(y_out, 1.0 - 1e-6)


def revivals_only_mapping(microscopic, taus_s, p):
    """
    Gate microscopic deviations to revivals AND add a zero-mean oscillatory term
    so the signal can go above baseline near revivals (as seen experimentally).

    p expects (in addition to your usual fine params):
      baseline, comb_contrast,
      revival_time (us), width0_us (us), T2_ms, T2_exp,
      amp_taper_alpha, width_slope, revival_chirp,
    """
    # ---- unpack ----
    baseline = float(p.get("baseline", 0.6))
    comb_contrast = float(p.get("comb_contrast", 0.4))
    Trev_us = max(1e-9, float(p.get("revival_time", 37.3)))
    w0_us = max(1e-9, float(p.get("width0_us", 6.0)))
    T2_ms = float(p.get("T2_ms", 0.08))
    T2_exp = float(p.get("T2_exp", 1.2))
    taper = float(p.get("amp_taper_alpha", 0.0))
    w_slope = float(p.get("width_slope", 0.0))
    chirp = float(p.get("revival_chirp", 0.0))
    # amplitude around 0d 0
    taus_us = np.asarray(taus_s, float) * 1e6
    # ---- comb mask (0..1), tightened by 'power' ----
    tau_max = float(np.nanmax(taus_us)) if taus_us.size else 0.0
    n_guess = max(1, min(64, int(np.ceil(1.2 * tau_max / Trev_us)) + 1))
    mask = _comb_quartic_powerlaw(
        taus_us, Trev_us, w0_us, taper, w_slope, chirp, n_guess
    )
    # microscopic factor m(τ) with m(0)=1
    m = np.asarray(microscopic, float)
    taus_us = np.asarray(taus_s, float) * 1e6  # x-axis in μs
    m = np.asarray(microscopic, float)

    # envelope
    T2_us = max(1e-9, 1000.0 * T2_ms)
    E = np.exp(-((taus_us / T2_us) ** T2_exp))
    # --- revival gate (≈0 at τ≈0, peaks at k*Trev) ---
    y_core = baseline - comb_contrast * m * mask * E
    # --- Physically motivated gains at revivals ---
    # A) charge pedestal
    # y_out = add_charge_pedestal(
    #     y_core, taus_us, mask, A_ch=0.03, T_ch_us=300.0, baseline=baseline
    # )
    # B) multiplicative gain
    # y_out = apply_readout_gain(y_core, mask, beta=0.08, baseline=baseline)
    return y_core


# =============================================================================
# Hyperfine handling
# =============================================================================
def _orthonormal_basis_from_z(z):
    """Build an ONB {ez=z/|z|, e1, e2}."""
    ez = np.asarray(z, float)
    ez /= np.linalg.norm(ez)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - ez * np.dot(tmp, ez)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    return ez, e1, e2


def make_R_NV(nv_axis_crystal):
    """
    nv_axis_crystal: tuple/list/np.array like (±1, ±1, ±1)
    Returns R such that v_NV = R @ v_crystal and ez_NV aligns with nv_axis.
    """
    n = np.asarray(nv_axis_crystal, float)
    n /= np.linalg.norm(n)  # unit
    # reuse your ONB constructor but for the NV axis now
    ez, e1, e2 = _orthonormal_basis_from_z(n)  # ez=n, e1,e2 ⟂ n
    # rows are basis vectors expressed in crystal coords → left-multiply for coords in NV basis
    return np.vstack([e1, e2, ez])


def compute_hyperfine_components(A_tensor, dir_hat):
    """
    Return (A_par, A_perp) in Hz, defined w.r.t. the *direction* dir_hat
    (usually the magnetic-field unit vector, not the NV axis).
    """
    ez = np.asarray(dir_hat, float)
    ez /= np.linalg.norm(ez)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(ez[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = tmp - ez * np.dot(tmp, ez)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ez, e1)
    A_par = ez @ A_tensor @ ez
    A_perp = np.sqrt((e1 @ A_tensor @ ez) ** 2 + (e2 @ A_tensor @ ez) ** 2)
    return A_par, A_perp


def U_111_to_cubic():
    ex = np.array([1.0, -1.0, 0.0])
    ex /= np.linalg.norm(ex)  # [1,-1,0]/√2
    ez = np.array([1.0, 1.0, 1.0])
    ez /= np.linalg.norm(ez)  # [1, 1,1]/√3
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)  # [1,1,-2]/√6
    # Columns are the file-frame basis vectors written in cubic coords
    U = np.column_stack([ex, ey, ez])  # maps components in 111-frame -> cubic
    return U


def A_file_to_cubic(A_file):
    U = U_111_to_cubic()
    return U @ A_file @ U.T


# =============================================================================
# Single-nucleus echo factor Mk(τ) and many-spin product
# =============================================================================
def Mk_tau(A_par_Hz, A_perp_Hz, omegaI_Hz, tau_s):
    """
    Exact single-nucleus ESEEM factor:
      ω_{-1} = sqrt((ωI - A_par)^2 + A_perp^2)
      ω_0    = ωI
      ω±     = ω_{-1} ± ω_0
      κ      = (A_perp^2) / (ω_{-1}^2)
      M(τ)   = 1 - κ * sin^2(π ω_+ τ) * sin^2(π ω_- τ)
    All freqs in Hz; τ in seconds.
    """
    wI = omegaI_Hz
    w_m1 = np.sqrt((wI - A_par_Hz) ** 2 + A_perp_Hz**2)
    w0 = wI
    w_plus = w_m1 + w0
    w_minus = w_m1 - w0
    kappa = (A_perp_Hz**2) / (w_m1**2 + 1e-30)

    # sin arguments need cycles → use π*freq*τ because sin^2(ω τ /2) with ω=2π f ⇒ sin^2(π f τ)
    return 1.0 - 2 * kappa * (np.sin(np.pi * w_plus * tau_s) ** 2) * (
        np.sin(np.pi * w_minus * tau_s) ** 2
    )


def compute_echo_signal(
    hyperfine_tensors,
    tau_array_s,
    B_field_vec_T,
    sigma_B_G=0.0,
    rng=None,
):
    B_vec = np.array(B_field_vec_T, float)
    if sigma_B_G > 0.0 and rng is not None:
        B_vec = B_vec + 1e-4 * rng.normal(0.0, sigma_B_G, size=3)  # G→T

    B_mag = np.linalg.norm(B_vec)
    if B_mag == 0.0:
        raise ValueError("B-field magnitude is zero.")
    B_unit = B_vec / B_mag
    omega_L = gamma_C13 * B_mag  # Hz

    signal = np.empty_like(tau_array_s, dtype=float)

    for i, tau in enumerate(tau_array_s):
        Mk_prod = 1.0
        for A_tensor in hyperfine_tensors:
            A_par, A_perp = compute_hyperfine_components(A_tensor, B_unit)
            Mk = Mk_tau(A_par, A_perp, omega_L, tau)
            Mk_prod *= Mk
        signal[i] = Mk_prod
    return signal


def Mk_from_catalog_rec(rec, tau_array_s):
    """
    ESEEM factor using (kappa, f0, f-1) from the catalog.

    rec must have:
      - 'kappa'
      - 'fI_Hz'       ≡ f0
      - 'omega_ms_Hz' ≡ f-1
    All freqs in Hz, tau_array_s in seconds.
    """

    kappa = float(rec["kappa"])
    f0 = float(rec["f0_Hz"])  # from fI_Hz
    f_m1 = float(rec["f_m1_Hz"])  # from omega_ms_Hz
    tau = np.asarray(tau_array_s, float)

    return 1.0 - 2.0 * kappa * (np.sin(np.pi * f0 * tau) ** 2) * (
        np.sin(np.pi * f_m1 * tau) ** 2
    )


def compute_echo_signal_from_catalog(chosen_sites, tau_array_s):
    signal = np.ones_like(tau_array_s, dtype=float)
    for rec in chosen_sites:
        signal *= Mk_from_catalog_rec(rec, tau_array_s)
    return signal


# =============================================================================
# RNG helpers and selection
# =============================================================================
def _spawn_streams(seed, num_streams, run_salt=None):
    if seed is None:
        ss = np.random.SeedSequence()
    else:
        if run_salt is None:
            ss = np.random.SeedSequence(int(seed))
        else:
            run_salt = int(run_salt) & 0xFFFFFFFF
            ss = np.random.SeedSequence(int(seed), spawn_key=[run_salt])
    child_seqs = ss.spawn(num_streams)
    return [np.random.default_rng(cs) for cs in child_seqs]


def _choose_sites(rng, present_sites, num_spins, selection_mode="uniform"):
    present_sites = list(present_sites)
    if num_spins is None or num_spins >= len(present_sites):
        return present_sites

    if selection_mode == "top_Apar":
        order = np.argsort([-abs(s["Apar_Hz"]) for s in present_sites])
        idx = order[:num_spins]

    elif selection_mode == "distance_weighted":
        r = np.array([max(s["dist"], 1e-12) for s in present_sites], float)
        w = 1.0 / (r**3)
        w /= w.sum()
        idx = np.sort(
            rng.choice(len(present_sites), size=num_spins, replace=False, p=w)
        )

    # NEW: pick strongest catalog entries by κ
    elif selection_mode == "top_kappa":
        order = np.argsort([-abs(s.get("kappa", 0.0)) for s in present_sites])
        idx = order[:num_spins]

    # NEW: pick by first-order line weight (minus+plus combined)
    elif selection_mode == "top_weight":
        order = np.argsort(
            [
                -(s.get("line_w_minus", 0.0) + s.get("line_w_plus", 0.0))
                for s in present_sites
            ]
        )
        idx = order[:num_spins]

    else:  # uniform
        idx = np.sort(rng.choice(len(present_sites), size=num_spins, replace=False))

    return [present_sites[i] for i in idx]


def read_hyperfine_table_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # find first data row that starts with an integer (skip headers/junk)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    def _is_int_start(s: str) -> bool:
        s = s.lstrip()
        if not s:
            return False
        t = s.split()[0]
        try:
            int(t)
            return True
        except Exception:
            return False

    try:
        data_start = next(i for i, line in enumerate(lines) if _is_int_start(line))
    except StopIteration:
        raise RuntimeError(f"Could not locate data start in hyperfine file: {path}")

    HF_COLS = [
        "index",
        "distance",
        "x",
        "y",
        "z",
        "Axx",
        "Ayy",
        "Azz",
        "Axy",
        "Axz",
        "Ayz",
    ]
    # primary path: pandas
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",  # robust whitespace split
            engine="python",
            comment="#",  # ignore commented tails
            header=None,
            names=HF_COLS,
            usecols=list(range(11)),  # ensure exactly 11 cols
            skiprows=data_start,
            na_filter=False,
        )
        # enforce dtypes
        df = df.astype(
            {
                "index": int,
                "distance": float,
                "x": float,
                "y": float,
                "z": float,
                "Axx": float,
                "Ayy": float,
                "Azz": float,
                "Axy": float,
                "Axz": float,
                "Ayz": float,
            },
            errors="ignore",
        )
        return df
    except Exception as e:
        # fallback: numpy → DataFrame
        arr = np.loadtxt(
            path,
            comments="#",
            dtype=float,
            ndmin=2,
        )
        if arr.shape[1] < 11:
            raise RuntimeError(
                f"Expected ≥11 columns, found {arr.shape[1]} in {path}"
            ) from e
        arr = arr[:, :11]
        df = pd.DataFrame(arr, columns=HF_COLS)
        # index is float now; coerce to int safely
        df["index"] = df["index"].round().astype(int)
        return df


# ---------- 0) IO ----------
def load_catalog(path_json: str) -> List[Dict]:
    """Load ESEEM catalog (with κ and f± in Hz) written by your builder."""
    with open(path_json, "r") as f:
        return json.load(f)


def _ori_tuple(o) -> Tuple[int, int, int]:
    return tuple(int(x) for x in o)


# =============================================================================
# Main simulator
# =============================================================================
def simulate_random_spin_echo_average(
    hyperfine_path,
    tau_range_us,
    num_spins=30,
    num_realizations=1,
    distance_cutoff=None,
    Ak_min_kHz=None,  # keep if A∥ ≥ Ak_min_kHz (if set)
    Ak_max_kHz=None,  # keep if A∥ ≤ Ak_max_kHz (if set)
    Ak_abs=True,  # compare |A∥| if True, signed A∥ if False
    R_NV=np.eye(3),
    fine_params=None,  # set None for microscopic-only
    abundance_fraction=0.011,
    rng_seed=None,
    run_salt=None,
    randomize_positions=False,  # keep False for single NV
    selection_mode="top_Apar",
    ensure_unique_across_realizations=False,  # usually False for fixed NV
    annotate_from_realization=0,
    keep_nv_orientation=True,  # keep True for single NV
    fixed_site_ids=None,  # exact sites to include
    fixed_presence_mask=None,  # boolean mask of length N_sites
    reuse_present_mask=True,  # draw Bernoulli once and reuse (quenched)
    # -------- NEW: catalog options --------
    catalog_json_path=None,  # if not None, use catalog instead of hyperfine_path
    nv_orientation=None,  # e.g. (1,1,1); if None, keep all orientations
):
    """
    Returns:
      taus_us, avg_signal, aux
      aux = {
        "positions": (N,3) of annotated realization (NV frame),
        "site_info": [{site_id, Apar_kHz, r}, ...],
        "revivals_us": array of k*revival_time for plotting,
        "picked_ids_per_realization": [[...], ...],
        "stats": {...}
      }
    """
    # Time axis
    taus_s = np.linspace(float(tau_range_us[0]), float(tau_range_us[1]), num=600) * 1e-6

    # RNG
    rng_streams = _spawn_streams(rng_seed, max(1, num_realizations), run_salt=run_salt)

    # Rotate B into NV frame once
    # Rotate B into NV frame once (still used by the hyperfine path)
    B_vec_NV = R_NV @ B_vec_T
    B_hat_NV = B_vec_NV / np.linalg.norm(B_vec_NV)

    # -------------------------------------------------------------------------
    # A) Use raw hyperfine table (original behavior)
    # -------------------------------------------------------------------------
    if catalog_json_path is None:
        df = read_hyperfine_table_safe(hyperfine_path)
        if distance_cutoff is not None:
            df = df[df["distance"] < distance_cutoff]

        sites = []
        for _, row in df.iterrows():
            A = (
                np.array(
                    [
                        [row.Axx, row.Axy, row.Axz],
                        [row.Axy, row.Ayy, row.Ayz],
                        [row.Axz, row.Ayz, row.Azz],
                    ],
                    float,
                )
                * 1e6
            )  # MHz -> Hz
            A_nv = R_NV @ A @ R_NV.T

            # Apparent A_parallel for current B (NV frame)
            A_par, _ = compute_hyperfine_components(A_nv, B_hat_NV)

            # A∥ filter (existing logic)
            A_par_kHz = (abs(A_par) if Ak_abs else A_par) / 1e3
            keep_A = True
            if Ak_min_kHz is not None:
                keep_A &= A_par_kHz >= float(Ak_min_kHz)
            if Ak_max_kHz is not None:
                keep_A &= A_par_kHz <= float(Ak_max_kHz)
            if not keep_A:
                continue

            pos_crystal = np.array([row.x, row.y, row.z], float)
            pos_nv = R_NV @ pos_crystal
            sites.append(
                {
                    "site_id": int(row["index"]),
                    "A0": A_nv,
                    "pos0": pos_nv,
                    "dist": float(row.distance),
                    "Apar_Hz": float(A_par),
                }
            )
    # -------------------------------------------------------------------------
    # B) Use pre-built ESEEM catalog (kappa, f± in lab frame)
    # -------------------------------------------------------------------------
    else:
        # Use pre-built ESEEM catalog (JSON list of dicts)
        catalog = load_catalog(catalog_json_path)

        # Optional: filter on NV orientation, e.g. (1,1,1)
        if nv_orientation is not None:
            ori_tgt = _ori_tuple(nv_orientation)
            catalog = [
                rec for rec in catalog if _ori_tuple(rec["orientation"]) == ori_tgt
            ]
            print(f"After orientation filter {ori_tgt}: {len(catalog)} records")

        # Optional: distance cutoff (Å)
        if distance_cutoff is not None:
            dmax = float(distance_cutoff)
            catalog = [rec for rec in catalog if rec.get("distance_A", np.inf) < dmax]
            print(f"After distance cutoff < {dmax} Å: {len(catalog)} records")

        f_minus_all = np.array([rec["f_minus_Hz"] for rec in catalog]) / 1e3
        f_plus_all = np.array([rec["f_plus_Hz"] for rec in catalog]) / 1e3
        print(f"f_- range (kHz): {f_minus_all.min():.1f} – {f_minus_all.max():.1f}")
        print(f"f_+ range (kHz): {f_plus_all.min():.1f} – {f_plus_all.max():.1f}")

        sites = []
        for rec in catalog:
            # -------- frequency-band filter using f_- and f_+ (in kHz) --------
            f_minus_Hz = float(rec["f_minus_Hz"])
            f_plus_Hz = float(rec["f_plus_Hz"])
            f_minus_kHz = f_minus_Hz / 1e3
            f_plus_kHz = f_plus_Hz / 1e3

            keep_f = True
            if (Ak_min_kHz is not None) or (Ak_max_kHz is not None):
                # Require *both* f_- and f_+ to lie inside [Ak_min_kHz, Ak_max_kHz]
                in_minus = True
                in_plus = True

                if Ak_min_kHz is not None:
                    lo = float(Ak_min_kHz)
                    in_minus &= f_minus_kHz >= lo
                    in_plus &= f_plus_kHz >= lo

                if Ak_max_kHz is not None:
                    hi = float(Ak_max_kHz)
                    in_minus &= f_minus_kHz <= hi
                    in_plus &= f_plus_kHz <= hi

                # keep only if BOTH lines are inside the band
                keep_f = in_minus and in_plus

            if not keep_f:
                continue

            # ------------------------------------------------------------------
            pos = np.array(
                [
                    rec.get("x_A", rec.get("x", np.nan)),
                    rec.get("y_A", rec.get("y", np.nan)),
                    rec.get("z_A", rec.get("z", np.nan)),
                ],
                float,
            )

            sites.append(
                {
                    "site_id": int(rec["site_index"]),
                    # keep keys expected by the rest of the code:
                    "A0": None,  # not used in catalog mode
                    "pos0": pos,
                    "dist": float(rec.get("distance_A", np.nan)),
                    # catalog-specific fields for time-domain M(τ)
                    "kappa": float(rec["kappa"]),
                    "f0_Hz": float(rec["fI_Hz"]),  # f_0  (ms = 0 manifold)
                    "f_m1_Hz": float(rec["omega_ms_Hz"]),  # f_-1 (ms = -1 manifold)
                    # sum/diff lines and weights
                    "f_minus_Hz": f_minus_Hz,
                    "f_plus_Hz": f_plus_Hz,
                    "line_w_minus": float(rec.get("line_w_minus", 0.0)),
                    "line_w_plus": float(rec.get("line_w_plus", 0.0)),
                }
            )

    N_candidates = len(sites)
    print(
        f"Total candidate sites after catalog filters "
        f"(orientation, distance, f_-/f_+ band): {N_candidates}"
    )
    if N_candidates == 0:
        taus_us = taus_s * 1e6
        flat = np.ones_like(taus_us)
        return (
            taus_us,
            flat,
            {
                "positions": None,
                "site_info": [],
                "revivals_us": None,
                "picked_ids_per_realization": [],
                "stats": {},
            },
        )

    id_to_idx = {s["site_id"]: i for i, s in enumerate(sites)}

    present_mask_global = None
    if (
        (fixed_site_ids is None)
        and (fixed_presence_mask is None)
        and reuse_present_mask
    ):
        rng_once = rng_streams[0]
        present_mask_global = rng_once.random(N_candidates) < abundance_fraction

    # Containers
    all_signals = []
    picked_ids_per_realization = []
    present_counts = []
    chosen_counts = []
    anno_positions = None
    anno_site_info = None
    anno_rev_times = None
    used_site_ids = set()

    for r in range(num_realizations):
        rng_r = rng_streams[r]

        # Decide occupancy
        if fixed_site_ids:
            present_idxs = np.array(
                [id_to_idx[i] for i in fixed_site_ids if i in id_to_idx], int
            )
            present_mask = np.zeros(N_candidates, dtype=bool)
            present_mask[present_idxs] = True
        elif fixed_presence_mask is not None:
            mask = np.asarray(fixed_presence_mask, bool)
            if mask.size != N_candidates:
                raise ValueError(
                    "fixed_presence_mask length does not match candidate site count."
                )
            present_mask = mask
            present_idxs = np.flatnonzero(present_mask)
        elif present_mask_global is not None:
            present_mask = present_mask_global
            present_idxs = np.flatnonzero(present_mask)
        else:
            present_mask = rng_r.random(N_candidates) < abundance_fraction
            present_idxs = np.flatnonzero(present_mask)

        present_counts.append(int(present_mask.sum()))
        if present_idxs.size == 0:
            all_signals.append(np.ones_like(taus_s))
            picked_ids_per_realization.append([])
            continue

        present_sites = [sites[i] for i in present_idxs]

        # Optional cross-realization uniqueness (usually False for fixed NV)
        if ensure_unique_across_realizations:
            filtered = [s for s in present_sites if s["site_id"] not in used_site_ids]
            if len(filtered) >= max(1, num_spins if num_spins is not None else 1):
                present_sites = filtered

        # Choose final subset (or take all)
        chosen_sites = _choose_sites(rng_r, present_sites, num_spins, selection_mode)
        chosen_counts.append(len(chosen_sites))
        picked_ids = [s["site_id"] for s in chosen_sites]
        picked_ids_per_realization.append(picked_ids)
        used_site_ids.update(picked_ids)

        # Compute echo: either from raw hyperfine tensors or from catalog
        if catalog_json_path is None:
            tensors = [s["A0"] for s in chosen_sites]
            signal = compute_echo_signal(tensors, taus_s, B_vec_NV)
        else:
            signal = compute_echo_signal_from_catalog(chosen_sites, taus_s)

        all_signals.append(signal)

        # Annotation (first realization by default)
        if r == annotate_from_realization:
            anno_positions = (
                np.array([s["pos0"] for s in chosen_sites]) if chosen_sites else None
            )

            if catalog_json_path is None:
                anno_site_info = [
                    {
                        "site_id": s["site_id"],
                        "Apar_kHz": float(
                            abs(compute_hyperfine_components(s["A0"], B_hat_NV)[0])
                            / 1e3
                        ),
                        "r": float(np.linalg.norm(s["pos0"])),
                    }
                    for s in chosen_sites
                ]
            else:
                # use catalog data directly (A∥ + f_- / f_+)
                anno_site_info = [
                    {
                        "site_id": s["site_id"],
                        "Apar_kHz": float(abs(s.get("Apar_Hz", 0.0)) / 1e3),
                        "r": float(np.linalg.norm(s["pos0"])),
                        "f_minus_kHz": float(s.get("f_minus_Hz", 0.0) / 1e3),
                        "f_plus_kHz": float(s.get("f_plus_Hz", 0.0) / 1e3),
                    }
                    for s in chosen_sites
                ]

            if fine_params is not None and "revival_time" in fine_params:
                revT_us = float(fine_params["revival_time"])
                kmax = int(np.ceil((taus_s.max() * 1e6) / revT_us))
                anno_rev_times = np.arange(0, kmax + 1) * revT_us

    # Average (for single realization this is just identity)
    avg_signal = np.mean(all_signals, axis=0)

    # phenomenological gating
    if fine_params is not None:
        if np.nanmax(avg_signal) - np.nanmin(avg_signal) < 1e-4:
            # no microscopic modulation -> synthesize clean comb at your baseline
            avg_signal = _synthesize_comb_only(taus_s, fine_params)
        else:
            # <-- key line: gate deviations so oscillations live only near revivals
            avg_signal = revivals_only_mapping(avg_signal, taus_s, fine_params)

    stats = {
        "N_candidates": N_candidates,
        "abundance_fraction": float(abundance_fraction),
        "present_counts": present_counts,
        "chosen_counts": chosen_counts,
    }

    return (
        taus_s * 1e6,
        avg_signal,
        {
            "positions": anno_positions,
            "site_info": anno_site_info if anno_site_info is not None else [],
            "revivals_us": anno_rev_times,
            "picked_ids_per_realization": picked_ids_per_realization,
            "stats": stats,
            "all_candidate_positions": np.array(
                [s["pos0"] for s in sites], float
            ),  # NEW
        },
    )


# =============================================================================
# Plotting
# =============================================================================
def set_axes_equal_3d(ax):
    """Make 3D axes have equal scale (so spheres look like spheres)."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])
    max_range = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(xmid - max_range, xmid + max_range)
    ax.set_ylim3d(ymid - max_range, ymid + max_range)
    ax.set_zlim3d(zmid - max_range, zmid + max_range)


def _echo_summary_lines(taus_us, echo):
    if len(echo) == 0:
        return []
    arr = np.asarray(echo, float)
    n = max(3, len(arr) // 3)
    early = float(np.nanmean(arr[:n]))
    late = float(np.nanmean(arr[-n:]))
    return [
        f"Echo range: {arr.min():.3f} … {arr.max():.3f}",
        f"⟨early⟩→⟨late⟩: {early:.3f} → {late:.3f}",
    ]


def _fine_param_lines(fine_params):
    if not fine_params:
        return []
    pretty = {
        "revival_time": "T_rev (μs)",
        "width0_us": "width₀ (μs)",
        "T2_ms": "T₂ (ms)",
        "T2_exp": "stretch n",
        "amp_taper_alpha": "amp taper α",
        "width_slope": "width slope",
        "revival_chirp": "rev chirp",
    }
    keys = [
        "revival_time",
        "width0_us",
        "T2_ms",
        "T2_exp",
        "amp_taper_alpha",
        "width_slope",
        "revival_chirp",
    ]
    out = []
    for k in keys:
        if k in fine_params:
            v = fine_params[k]
            sval = f"{v:.3g}" if isinstance(v, (int, float)) else f"{v}"
            out.append(f"{pretty[k]}: {sval}")
    return out


# def _site_table_lines(site_info, max_rows=8):
#     if not site_info:
#         return ["(no annotated realization)"]
#     rows = sorted(site_info, key=lambda d: -abs(d.get("Apar_kHz", 0.0)))
#     lines = ["site  |A∥|(kHz)   r", "------------------------"]
#     for d in rows[:max_rows]:
#         sid = d.get("site_id", "?")
#         apar = float(abs(d.get("Apar_kHz", 0.0)))
#         rmag = float(d.get("r", np.nan))
#         lines.append(f"{sid:<5} {apar:>8.0f}  {rmag:>6.2f}")
#     if len(rows) > max_rows:
#         lines.append(f"... (+{len(rows)-max_rows} more)")
#     return lines
#
def _site_table_lines(site_info, max_rows=8):
    """
    Build a small text table for the annotation box on the 3D panel.

    For catalog mode we show:
      site_id | r (Å) | f_- (kHz) | f_+ (kHz)
    """
    if not site_info:
        return ["No sites selected"]

    lines = []
    lines.append("id   r(Å)   f_-(kHz)   f_+(kHz)")

    n = min(len(site_info), max_rows)
    for meta in site_info[:n]:
        sid = meta.get("site_id", "?")
        r = float(meta.get("r", np.nan))
        f_m = float(meta.get("f_minus_kHz", np.nan))
        f_p = float(meta.get("f_plus_kHz", np.nan))

        lines.append(f"{sid:3d}  {r:5.2f}   {f_m:8.1f}   {f_p:8.1f}")

    if len(site_info) > max_rows:
        lines.append(f"... (+{len(site_info) - max_rows} more)")

    return lines


def _env_only_curve(taus_us, fine_params):
    """baseline - envelope(τ); ignores COMB/MOD so you see pure T2 envelope."""
    if not fine_params:
        return None
    baseline = float(fine_params.get("baseline", 1.0))
    T2_ms = float(fine_params.get("T2_ms", 1.0))
    T2_exp = float(fine_params.get("T2_exp", 1.0))
    # envelope(τ) = exp[-(τ/(1000*T2_ms))^T2_exp]
    env = np.exp(-((np.asarray(taus_us, float) / (1000.0 * T2_ms)) ** T2_exp))
    # multiply by comb_contrast if you want to visualize the amplitude scale
    contrast = float(fine_params.get("comb_contrast", 1.0))
    return baseline - contrast * env


def _comb_only_curve(taus_us, fine_params):
    """
    Very light-weight comb sketch (Gaussian revivals); ignores oscillations and width slope.
    Useful if you want to also show envelope×comb (set show_env_times_comb=True).
    """
    if not fine_params:
        return None
    T_rev = float(
        fine_params.get("revival_time", fine_params.get("revival_time_us", 0.0))
    )
    width0 = float(fine_params.get("width0_us", 0.0))
    alpha = float(fine_params.get("amp_taper_alpha", 0.0))
    if T_rev <= 0 or width0 <= 0:
        return np.ones_like(taus_us, dtype=float)

    τ = np.asarray(taus_us, float)
    mmax = int(max(1, np.ceil(τ.max() / T_rev) + 2))
    comb = np.zeros_like(τ, float)
    # sum of Gaussians centered at m*T_rev with amplitude taper ~ exp(-alpha*m)
    for m in range(mmax + 1):
        amp = np.exp(-alpha * m) if alpha > 0 else 1.0
        comb += amp * np.exp(-0.5 * ((τ - m * T_rev) / width0) ** 2)
    # normalize to [0,1] peak
    mx = comb.max()
    if mx > 0:
        comb = comb / mx
    return comb


def plot_echo_with_sites(
    taus_us,
    echo,
    aux,
    title="Spin Echo (single NV)",
    rmax=None,
    fine_params=None,
    units_label="(arb units)",
    nv_label=None,  # <-- NEW: show NV id
    nv_orientation=None,
    sim_info=None,  # <-- NEW: dict with sim settings to display
    show_env=True,  # <-- NEW: overlay envelope-only
    show_env_times_comb=False,  # <-- NEW: optionally overlay envelope×comb
):
    fig = plt.figure(figsize=(12, 5))

    # ---------------- Echo panel ----------------
    ax0 = fig.add_subplot(1, 2, 1)
    ax0.plot(taus_us, echo, lw=1.0, label="echo")
    ax0.set_xlabel("τ (μs)")
    ax0.set_ylabel(f"Coherence {units_label}")

    # Title: include NV label if provided
    if nv_label and nv_orientation is not None:
        ax0.set_title(f"{title} — NV {nv_label} [{nv_orientation}]")
    elif nv_label is not None and nv_orientation is None:
        ax0.set_title(f"{title} — NV {nv_label}")
    else:
        ax0.set_title(title)

    ax0.grid(True, alpha=0.3)

    # Vertical revival guide lines (if provided)
    revs = aux.get("revivals_us", None)
    if revs is not None:
        for t in np.atleast_1d(revs):
            ax0.axvline(t, ls="--", lw=0.7, alpha=0.35)

    # --- NEW: overlay envelope(s) ---
    env_line = None
    if show_env and fine_params:
        y_env = _env_only_curve(taus_us, fine_params)
        if y_env is not None:
            (env_line,) = ax0.plot(
                taus_us, y_env, ls="--", lw=1.2, label="envelope (T₂)", alpha=0.9
            )

    if show_env_times_comb and fine_params:
        comb = _comb_only_curve(taus_us, fine_params)
        if comb is not None:
            baseline = float(fine_params.get("baseline", 1.0))
            contrast = float(fine_params.get("comb_contrast", 1.0))
            T2_ms = float(fine_params.get("T2_ms", 1.0))
            T2_exp = float(fine_params.get("T2_exp", 1.0))
            env = np.exp(-((np.asarray(taus_us, float) / (1000.0 * T2_ms)) ** T2_exp))
            y_env_comb = baseline - contrast * env * comb
            ax0.plot(
                taus_us,
                y_env_comb,
                ls=":",
                lw=1.2,
                label="envelope×comb (no osc)",
                alpha=0.9,
            )

    # Existing stats box
    stats = aux.get("stats", {}) or {}
    # lines_stats = []
    # if "N_candidates" in stats:
    #     lines_stats.append(f"Candidates: {stats['N_candidates']}")
    # if "abundance_fraction" in stats:
    #     lines_stats.append(f"Abundance p: {100*stats['abundance_fraction']:.2f}%")
    # if "chosen_counts" in stats and stats["chosen_counts"]:
    #     cc = np.asarray(stats["chosen_counts"], int)
    #     lines_stats.append(f"Chosen/site per realization: {int(np.median(cc))} (med)")
    # if lines_stats:
    #     ax0.text(
    #         0.61,
    #         0.02,
    #         "\n".join(lines_stats),
    #         transform=ax0.transAxes,
    #         fontsize=9,
    #         va="bottom",
    #         ha="left",
    #         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.6, lw=0.6),
    #     )

    # Fine-parameter box (existing)
    # ---- Combined NV/sim + fine-params box (single box, right-top) ----
    combined_lines = []

    # Header & flags
    if nv_label is not None:
        flag_bits = []
        if show_env:
            flag_bits.append("Env")
        if show_env_times_comb:
            flag_bits.append("Comb")
        hdr = f"NV: {nv_label}"
        if flag_bits:
            hdr += f"  [{'+'.join(flag_bits)} shown]"
        combined_lines.append(hdr)

    # Build a meta dict from sim_info with fallbacks to aux
    meta = {} if sim_info is None else dict(sim_info)
    # meta.setdefault("selection_mode", aux.get("selection_mode"))
    meta.setdefault("distance_cutoff", aux.get("distance_cutoff"))
    meta.setdefault("Ak_min_kHz", aux.get("Ak_min_kHz"))
    meta.setdefault("Ak_max_kHz", aux.get("Ak_max_kHz"))
    # meta.setdefault("Ak_abs", aux.get("Ak_abs"))
    # meta.setdefault("reuse_present_mask", aux.get("reuse_present_mask"))
    # meta.setdefault("hyperfine_path", aux.get("hyperfine_path"))
    meta.setdefault("T2_fit_us", None)  # you can set this upstream if desired

    # Pretty labels
    pretty_sim = {
        # "selection_mode": "select",
        "distance_cutoff": "d_cut (Å)",
        "Ak_min_kHz": "Ak_min (kHz)",
        "Ak_max_kHz": "Ak_max (kHz)",
        # "Ak_abs": "Ak|·|?",
        # "reuse_present_mask": "reuse mask?",
        # "hyperfine_path": "HF",
        "T2_fit_us": "T2_fit (μs)",
    }

    def _fmt_meta(k, v):
        if v is None:
            return None
        lab = pretty_sim.get(k, k)
        if k == "hyperfine_path":
            from pathlib import Path

            v = Path(str(v)).stem
        if isinstance(v, float):
            # compact floats
            v = f"{v:.3g}"
        return f"{lab}: {v}"

    # Collect sim/meta lines (only those that exist)
    sim_lines = []
    for k in [
        # "selection_mode",
        "distance_cutoff",
        "Ak_min_kHz",
        "Ak_max_kHz",
        # "Ak_abs",
        # "reuse_present_mask",
        # "hyperfine_path",
        "T2_fit_us",
    ]:
        line = _fmt_meta(k, meta.get(k))
        if line:
            sim_lines.append(line)

    # Fine-parameter lines
    fp_lines = _fine_param_lines(fine_params) if fine_params else []
    if fp_lines and show_env:
        fp_lines = ["Exp Params."] + fp_lines

    # Merge sections with a thin separator if both present
    if sim_lines and fp_lines:
        combined_lines.extend(sim_lines + ["—"] + fp_lines)
    elif sim_lines:
        combined_lines.extend(sim_lines)
    elif fp_lines:
        combined_lines.extend(fp_lines)

    # Render the single box (right-top)
    # if combined_lines:
    #     ax0.text(
    #         0.99,
    #         0.02,
    #         "\n".join(combined_lines),
    #         transform=ax0.transAxes,
    #         fontsize=9,
    #         va="bottom",
    #         ha="right",
    #         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.5, lw=0.6),
    #     )

    # Legend if we drew extra curves
    if (show_env and fine_params) or show_env_times_comb:
        ax0.legend(loc="best", fontsize=9, framealpha=0.8)

    # ---------------- 3D positions panel ----------------
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    bg = aux.get("all_candidate_positions", None)
    if bg is not None and len(bg) > 0:
        ax1.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=8, alpha=0.15)

    pos = aux.get("positions", None)
    info = aux.get("site_info", [])
    if pos is not None and len(pos) > 0:
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20, depthshade=True)
        # for pnt, meta in zip(pos, info):
        #     # sid = meta.get("site_id", "?")
        #     # apar = meta.get("Apar_kHz", 0.0)
        #     rmag = meta.get("r", np.nan)
        #     # label = f"{sid}\n|A∥|={apar:.0f} kHz\nr={rmag:.2f}"
        #     label = f"r={rmag:.2f}"
        #     # label = f"{sid}"
        #     ax1.text(pnt[0], pnt[1], pnt[2], label, fontsize=8, ha="left", va="bottom")
        for pnt, meta in zip(pos, info):
            rmag = meta.get("r", np.nan)
            f_m = meta.get("f_minus_kHz", np.nan)
            label = f"r={rmag:.2f}\nf_-={f_m:.0f} kHz"
            ax1.text(pnt[0], pnt[1], pnt[2], label, fontsize=7, ha="left", va="bottom")

    ax1.scatter([0], [0], [0], s=70, marker="*", zorder=5)
    ax1.text(0, 0, 0, "NV", fontsize=9, ha="right", va="top")
    ax1.set_title("¹³C positions (NV frame)")
    ax1.set_xlabel("x (Å)")
    ax1.set_ylabel("y (Å)")
    ax1.set_zlabel("z (Å)")

    if rmax is None:
        if bg is not None and len(bg) > 0:
            rmax = float(np.max(np.linalg.norm(bg, axis=1)))
        elif pos is not None and len(pos) > 0:
            rmax = float(np.max(np.linalg.norm(pos, axis=1)))
        else:
            rmax = 1.0
    rpad = 0.05 * rmax
    ax1.set_xlim(-rmax - rpad, rmax + rpad)
    ax1.set_ylim(-rmax - rpad, rmax + rpad)
    ax1.set_zlim(-rmax - rpad, rmax + rpad)
    set_axes_equal_3d(ax1)

    picked_all = aux.get("picked_ids_per_realization", [])
    n_real = len(picked_all) if picked_all is not None else 0
    n_chosen = len(info) if info is not None else 0
    left_box = [f"Chosen Sites: {n_chosen}", f"Realizations: {n_real}"]
    if stats.get("N_candidates") is not None:
        left_box.append(f"Candidates: {stats['N_candidates']}")
    if "abundance_fraction" in stats:
        left_box.append(f"Abundance p: {100*stats['abundance_fraction']:.2f}%")
    ax1.text2D(
        0.01,
        0.02,
        "\n".join(left_box),
        transform=ax1.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.6),
    )

    table_lines = _site_table_lines(info, max_rows=8)
    ax1.text2D(
        0.99,
        0.02,
        "\n".join(table_lines),
        transform=ax1.transAxes,
        fontsize=9,
        # family="monospace",
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.8, lw=0.6),
    )

    # kpl.show()
    return fig


def _site_kappa_max(
    A_tensor_Hz: np.ndarray, B_hat_NV: np.ndarray, omegaI_Hz: float
) -> float:
    """
    Best-case single-nucleus ESEEM amplitude κ for a site, given B direction (NV frame).
    """
    # Decompose relative to NV axis (B_hat_NV is fine—NV axis aligned with B in your solver step)
    A_par, A_perp = compute_hyperfine_components(A_tensor_Hz, B_hat_NV)
    w_m1 = np.sqrt((omegaI_Hz - A_par) ** 2 + A_perp**2)
    # κ = (A_perp^2) / (w_m1^2)
    return float(A_par)
    # return float((A_perp * A_perp) / (w_m1 * w_m1 + 1e-30))


def plot_Apar_both_projections(
    hyperfine_path: str,
    orientations=((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)),
    distance_max: float | None = None,
):
    df = read_hyperfine_table_safe(hyperfine_path)
    if distance_max is not None:
        df = df[df["distance"] <= float(distance_max)].copy()

    colors = ["C0", "C1", "C2", "C3"]
    plt.figure(figsize=(8, 5))

    for ax, c in zip(orientations, colors):
        R = make_R_NV(ax)
        # B in this NV frame
        B_NV = R @ np.asarray(B_vec_T, float)
        B_hat_NV = B_NV / np.linalg.norm(B_NV)

        # Collect |A_par| for both projections
        dist, Apar_B_kHz, Apar_NV_kHz = [], [], []
        for _, row in df.iterrows():
            A_cr = (
                np.array(
                    [
                        [row.Axx, row.Axy, row.Axz],
                        [row.Axy, row.Ayy, row.Ayz],
                        [row.Axz, row.Ayz, row.Azz],
                    ],
                    float,
                )
                * 1e6
            )  # MHz→Hz
            A_nv = R @ A_cr @ R.T

            # proj along B (orientation-invariant)
            A_par_B = float(B_hat_NV @ A_nv @ B_hat_NV) / 1e3  # Hz→kHz
            # proj along NV axis (orientation-dependent)
            ez_NV = np.array([0.0, 0.0, 1.0])
            A_par_NV = float(ez_NV @ A_nv @ ez_NV) / 1e3

            dist.append(row.distance)
            Apar_B_kHz.append(abs(A_par_B))
            Apar_NV_kHz.append(abs(A_par_NV))

        # sort by distance for clean curves
        idx = np.argsort(dist)
        d = np.asarray(dist)[idx]
        aB = np.asarray(Apar_B_kHz)[idx]
        aN = np.asarray(Apar_NV_kHz)[idx]

        # plot: B-projection = solid, NV-projection = dashed (same color)
        plt.plot(d, aB, "-", lw=1.3, c=c, label=f"{ax} • |A∥| along B")
        plt.plot(d, aN, "--", lw=1.3, c=c, label=f"{ax} • |A∥| along NV")

    plt.yscale("log")
    plt.xlabel("Site distance (table units)")
    plt.ylabel(r"$|A_{\parallel}|$ (kHz)")
    plt.title(r"$|A_{\parallel}|$ vs distance — solid: proj(B), dashed: proj(NV)")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Example usage
# =============================================================================

# --- 0) Pull your saved fit dict (as you outlined) ---
# file_stem = "2025_11_02-19_55_17-johnson_204nv_s3-003c56"
file_stem = "2025_11_17-09_49_42-sample_204nv_s1-fcc605"
fit = dm.get_raw_data(file_stem=file_stem)

keys = fit["unified_keys"]


def _asdict(p):
    d = {k: None for k in keys}
    if p is None:
        return d
    for k, v in zip(keys, p + [None] * (len(keys) - len(p))):
        d[k] = v
    return d


labels = list(map(int, fit["nv_labels"]))
nv_orientations = [tuple(o) for o in fit["orientations"]]
popts = fit["popts"]
chis = fit.get("red_chi2", [None] * len(popts))

# --- 1) Make a clean parameter matrix (NaNs for missing), in unified_keys order ---
P = np.full((len(popts), len(keys)), np.nan, float)
for i, p in enumerate(popts):
    d = _asdict(p)
    for j, k in enumerate(keys):
        v = d[k]
        if v is None:
            continue
        try:
            P[i, j] = float(v)
        except Exception:
            pass

# Convenience indices
K = {k: j for j, k in enumerate(keys)}
k_base = K["baseline"]
k_cc = K["comb_contrast"]
k_Trev = K["revival_time_us"]
k_w0 = K["width0_us"]
k_T2 = K["T2_ms"]
k_n = K["T2_exp"]
k_a = K["amp_taper_alpha"]
k_ws = K["width_slope"]
k_ch = K["revival_chirp"]
k_A = K["osc_amp"]
k_f0 = K["osc_f0"]
k_f1 = K["osc_f1"]
k_p0 = K["osc_phi0"]
k_p1 = K["osc_phi1"]


# --- 2) Build cohort priors (median & MAD) for each parameter ---
def _nanmedian(x):
    return np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else np.nan


def _mad(x):
    med = _nanmedian(x)
    if not np.isfinite(med):
        return np.nan
    return _nanmedian(np.abs(x - med)) * 1.4826


cohort_med = np.array([_nanmedian(P[:, j]) for j in range(P.shape[1])])
cohort_mad = np.array([_mad(P[:, j]) for j in range(P.shape[1])])

# Reasonable hard bounds/guards (edit to taste)
bounds = {
    "baseline": (0.0, 1.2),
    "comb_contrast": (0.0, 1.1),
    "revival_time_us": (25.0, 55.0),
    "width0_us": (1.0, 25.0),
    "T2_ms": (1e-3, 2e3),
    "T2_exp": (0.6, 4.0),
    "amp_taper_alpha": (0.0, 2.0),
    "width_slope": (-0.2, 0.2),
    "revival_chirp": (-0.06, 0.06),
    "osc_amp": (-0.3, 0.3),  # allow above-baseline crests
    "osc_f0": (0.0, 0.50),  # cycles/μs ~ MHz
    "osc_f1": (0.0, 0.50),
    "osc_phi0": (-np.pi, np.pi),
    "osc_phi1": (-np.pi, np.pi),
}


def _clip_by_key(k, v):
    lo, hi = bounds[k]
    return float(np.minimum(hi, np.maximum(lo, v)))


# --- 3) NV-specific jitter scale (use MAD around each NV if you have repeats; else cohort MAD) ---
# Here we’ll just use cohort MAD and a mixing factor so we don’t over-jitter.
mix_global = 0.3  # 30% cohort prior noise
rng = np.random.default_rng(20251102)


def _nv_prior_draw(i):
    out = {}
    for k, j in K.items():
        # center at NV’s fitted value if finite; else cohort median
        mu = P[i, j] if np.isfinite(P[i, j]) else cohort_med[j]
        if not np.isfinite(mu):
            # final fallback constants
            if k == "baseline":
                mu = 0.6
            elif k == "comb_contrast":
                mu = 0.45
            elif k == "revival_time_us":
                mu = 38.0
            elif k == "width0_us":
                mu = 7.0
            elif k == "T2_ms":
                mu = 0.08
            elif k == "T2_exp":
                mu = 1.2
            else:
                mu = 0.0

        # jitter σ: a blend of cohort MAD and a small absolute floor
        sig = cohort_mad[j]
        if not np.isfinite(sig) or sig == 0.0:
            sig = 1e-3
        # symmetric jitter for all (including phases); widen a bit for osc_amp
        widen = 1.0 if k != "osc_amp" else 1.5
        draw = mu + widen * mix_global * sig * rng.standard_normal()

        out[k] = _clip_by_key(k, draw)
    return out


# --- 4) Run stochastic synthetic echoes per NV, then aggregate ---
def synth_per_nv(
    nv_idx,
    R=8,
    tau_range_us=(0, 100),
    f_range_kHz=(10, 1500),
    hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
    abundance_fraction=0.011,
    distance_cutoff=8.0,
    num_spins=None,  # None = all present, else subsample
    selection_mode="top_Apar",  # or "uniform" / "distance_weighted"
    reuse_present_mask=True,
    catalog_json_path=None,
):
    lbl = int(labels[nv_idx])
    nv_ori = nv_orientations[nv_idx]
    Ak_min_kHz, Ak_max_kHz = f_range_kHz

    salt = lbl & 0xFFFFFFFF

    traces = []  # list of echo arrays (length T)
    fine_list = []  # list of fine_params dicts per realization
    taus_ref = None
    aux_first = None
    fine_first = None
    echo_first = None

    for r in range(R):
        fp = _nv_prior_draw(nv_idx)
        fine_params = dict(
            baseline=fp["baseline"],
            comb_contrast=fp["comb_contrast"],
            revival_time=fp["revival_time_us"],
            width0_us=fp["width0_us"],
            T2_ms=fp["T2_ms"],
            T2_exp=fp["T2_exp"],
            amp_taper_alpha=fp["amp_taper_alpha"],
            width_slope=fp["width_slope"],
            revival_chirp=fp["revival_chirp"],
            # Additive oscillations (MHz == cycles/us)
            osc_add_amp=fp["osc_amp"],
            osc_f0_MHz=fp["osc_f0"],
            osc_f1_MHz=fp["osc_f1"],
            osc_phi0=fp["osc_phi0"],
            osc_phi1=fp["osc_phi1"],
        )

        taus, echo, aux = simulate_random_spin_echo_average(
            hyperfine_path=hyperfine_path,
            tau_range_us=tau_range_us,
            num_spins=num_spins,
            num_realizations=1,
            distance_cutoff=distance_cutoff,  # <= use the function arg
            Ak_min_kHz=Ak_min_kHz,
            Ak_max_kHz=Ak_max_kHz,
            Ak_abs=True,
            R_NV=np.eye(3),
            fine_params=fine_params,
            abundance_fraction=abundance_fraction,
            rng_seed=4242,
            run_salt=salt + r,
            randomize_positions=False,
            selection_mode=selection_mode,
            ensure_unique_across_realizations=False,
            annotate_from_realization=0,
            keep_nv_orientation=True,
            fixed_site_ids=None,
            fixed_presence_mask=None,
            reuse_present_mask=reuse_present_mask,
            # NEW: catalog usage
            catalog_json_path=catalog_json_path,
            nv_orientation=nv_ori,  # NV orientation for this NV
        )

        # (Optional) Noise model here if desired
        # echo = apply_noise_pipeline(echo, taus, rng, NOISE)
        # --- Plot single-NV detail using the FIRST realization payload ---
        plot_echo_with_sites(
            taus,
            echo,
            aux,
            fine_params=fine_params,
            nv_label=lbl,
            nv_orientation=nv_ori,
            sim_info=dict(
                selection_mode=selection_mode,
                distance_cutoff=distance_cutoff,
                Ak_min_kHz=Ak_min_kHz,
                Ak_max_kHz=Ak_max_kHz,
                Ak_abs=True,
                reuse_present_mask=reuse_present_mask,
                hyperfine_path=hyperfine_path,
                abundance_fraction=abundance_fraction,
                num_spins=("all" if num_spins is None else int(num_spins)),
            ),
            show_env=False,
            show_env_times_comb=False,
        )
        plt.show()

        if taus_ref is None:
            taus_ref = np.asarray(taus, float)

        traces.append(np.asarray(echo, float))
        fine_list.append(fine_params)

        if r == 0:
            aux_first = aux
            fine_first = fine_params
            echo_first = np.asarray(echo, float)

    traces = np.asarray(traces, float)  # [R, T]
    mean = np.nanmean(traces, axis=0)
    p16 = np.nanpercentile(traces, 16, axis=0)
    p84 = np.nanpercentile(traces, 84, axis=0)

    # Pack useful metadata (some pulled from aux of first realization)
    sim_info = dict(
        selection_mode=selection_mode,
        distance_cutoff=distance_cutoff,
        Ak_min_kHz=0,
        Ak_max_kHz=600,
        Ak_abs=True,
        reuse_present_mask=reuse_present_mask,
        hyperfine_path=hyperfine_path,
        abundance_fraction=abundance_fraction,
        num_spins=("all" if num_spins is None else int(num_spins)),
    )

    return dict(
        label=lbl,
        taus_us=taus_ref,
        mean=mean,
        p16=p16,
        p84=p84,
        fine_draws=fine_list,  # all draws
        # first-realization payload for detailed plotting:
        first_echo=echo_first,
        aux_first=aux_first,
        fine_params_first=fine_first,
        sim_info=sim_info,
    )


# --- 5) Orchestrate over a subset / all NVs ---
def _to_float_or_inf(x):
    try:
        if x is None:
            return np.inf
        v = float(x)
        return v if np.isfinite(v) else np.inf
    except Exception:
        return np.inf


def main():
    # ---- Config ----
    CONTRAST_MIN = 0.10  # keep NVs with fitted comb_contrast >= 0.10
    CONTRAST_MAX = 2.00  # e.g., set to 0.80 if you also want an upper cap

    # ---- Columns ----
    cc = P[:, k_cc]  # comb_contrast from your fits (dimensionless, typically 0..1)

    # ---- Masks ----
    has_cc = np.isfinite(cc)
    meets_cc = has_cc & (abs(cc) >= CONTRAST_MIN)
    if CONTRAST_MAX is not None and np.isfinite(CONTRAST_MAX):
        meets_cc &= abs(cc) <= CONTRAST_MAX

    # ---- Combine with your existing filters ----
    # (example: T2_us threshold + chi2 ordering)
    T2_ms = P[:, k_T2]
    T2_us = T2_ms * 1000.0
    T2_THRESHOLD_US = 2000.0
    fast_T2 = np.isfinite(T2_us) & (T2_us < T2_THRESHOLD_US)

    chi_vals = np.array([_to_float_or_inf(c) for c in chis], float)
    order = np.argsort(chi_vals)  # lowest χ² first
    has_good_chi = np.isfinite(chi_vals)  # or add your own chi2 cap

    filtered = [
        i
        for i in order
        if (popts[i] is not None and has_good_chi[i] and fast_T2[i] and meets_cc[i])
    ]
    keep = filtered[:15]
    print(
        f"Selected {len(keep)} NVs with contrast ≥ {CONTRAST_MIN}"
        + (f" and ≤ {CONTRAST_MAX}" if CONTRAST_MAX is not None else "")
        + f", T2 < {T2_THRESHOLD_US} µs."
    )

    results = []
    for i in keep:
        res = synth_per_nv(
            i,
            R=1,
            tau_range_us=(0, 100),
            f_range_kHz=(10, 1500),
            hyperfine_path="analysis/nv_hyperfine_coupling/nv-2.txt",
            num_spins=1,
            selection_mode="top_kappa",
            distance_cutoff=22.0,
            catalog_json_path="analysis\spin_echo_work\essem_freq_kappa_catalog_22A_49G.json",
        )
        results.append(res)
    # You now have a list of per-NV aggregates in `results`.
    return results


if __name__ == "__main__":
    kpl.init_kplotlib()
    # results = main()
    plt.show(block=True)
