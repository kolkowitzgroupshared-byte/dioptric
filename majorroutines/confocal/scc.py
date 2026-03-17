import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Adjustable parameters
# ----------------------------
N = 200_000  # number of shots to simulate per spin state

# Conventional readout: very low detected photons per shot
# (typical widefield numbers can be ~0.01–0.1 photons/shot per NV per readout window)
alpha0_conv = 0.03  # mean photons if ms=0
alpha1_conv = 0.02  # mean photons if ms=1

# SCC model = (spin -> charge mapping) + (charge photon readout)
# Mapping: probability the final charge is NV- depends on spin state
p_minus_ms0 = 0.85
p_minus_ms1 = 0.55

# Charge readout: photon means for NV- (bright) and NV0 (dim)
mu_minus = 4.0  # mean photons if NV-
mu_zero = 0.6  # mean photons if NV0


# ----------------------------
# Helpers
# ----------------------------
def simulate_conventional(N, a0, a1, rng):
    n0 = rng.poisson(a0, size=N)
    n1 = rng.poisson(a1, size=N)
    return n0, n1


def simulate_scc(N, pminus0, pminus1, mu_minus, mu_zero, rng):
    # For each spin, draw whether it becomes NV- or NV0, then photon counts accordingly
    charge0 = rng.random(N) < pminus0  # True => NV-
    charge1 = rng.random(N) < pminus1
    n0 = rng.poisson(mu_minus, size=N) * charge0 + rng.poisson(mu_zero, size=N) * (
        ~charge0
    )
    n1 = rng.poisson(mu_minus, size=N) * charge1 + rng.poisson(mu_zero, size=N) * (
        ~charge1
    )
    return n0, n1


def best_threshold_fidelity(n0, n1):
    """
    Find integer threshold k such that we decide ms=0 if n >= k, else ms=1,
    maximizing average accuracy (equal priors).
    Returns: best_k, fidelity, confusion rates
    """
    max_n = int(max(n0.max(), n1.max()))
    best = (-1, -1.0, None)
    for k in range(max_n + 2):  # include k = max_n+1
        # decision rule: predict ms=0 if n >= k else ms=1
        acc0 = np.mean(n0 >= k)  # correct when true ms=0
        acc1 = np.mean(n1 < k)  # correct when true ms=1
        fidelity = 0.5 * (acc0 + acc1)
        if fidelity > best[1]:
            best = (k, fidelity, (acc0, acc1))
    return best  # (k, fidelity, (acc0, acc1))


def readout_snr_like(n0, n1):
    """
    A quick Poisson-inspired metric often used for NV readout:
    (mean0-mean1)/sqrt(mean0+mean1)
    Even for SCC (non-Poisson), it’s still a useful separability proxy.
    """
    m0, m1 = np.mean(n0), np.mean(n1)
    return (m0 - m1) / np.sqrt(m0 + m1 + 1e-12)


def sigma_R_from_snr(SNR):
    # Common NV readout-noise proxy: sigma_R = sqrt(1 + 2/SNR^2)
    return np.sqrt(1.0 + 2.0 / (SNR**2 + 1e-12))


# ----------------------------
# Run simulation
# ----------------------------
rng = np.random.default_rng(0)

n0c, n1c = simulate_conventional(N, alpha0_conv, alpha1_conv, rng)
n0s, n1s = simulate_scc(N, p_minus_ms0, p_minus_ms1, mu_minus, mu_zero, rng)

k_c, F_c, (acc0_c, acc1_c) = best_threshold_fidelity(n0c, n1c)
k_s, F_s, (acc0_s, acc1_s) = best_threshold_fidelity(n0s, n1s)

snr_c = readout_snr_like(n0c, n1c)
snr_s = readout_snr_like(n0s, n1s)

print("=== Conventional readout ===")
print(f"means: ms0={np.mean(n0c):.4f}, ms1={np.mean(n1c):.4f}")
print(f"SNR~ {snr_c:.3f}  -> sigma_R~ {sigma_R_from_snr(snr_c):.2f}")
print(
    f"best threshold k={k_c}, fidelity={F_c:.3f}  (acc0={acc0_c:.3f}, acc1={acc1_c:.3f})"
)

print("\n=== SCC readout ===")
print(f"means: ms0={np.mean(n0s):.3f}, ms1={np.mean(n1s):.3f}")
print(f"SNR~ {snr_s:.3f}  -> sigma_R~ {sigma_R_from_snr(snr_s):.2f}")
print(
    f"best threshold k={k_s}, fidelity={F_s:.3f}  (acc0={acc0_s:.3f}, acc1={acc1_s:.3f})"
)

# ----------------------------
# Plot histograms
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Conventional (use a small range; almost all mass is at 0 or 1)
bins_c = np.arange(0, 6) - 0.5
axes[0].hist(n0c, bins=bins_c, density=True, alpha=0.6, label="ms=0")
axes[0].hist(n1c, bins=bins_c, density=True, alpha=0.6, label="ms=1")
axes[0].axvline(k_c - 0.5, linestyle="--")
axes[0].set_title("Conventional readout (very low photons)")
axes[0].set_xlabel("Detected photons / shot")
axes[0].set_ylabel("Probability")
axes[0].set_yscale("log")
axes[0].legend()

# SCC (broader)
max_plot = int(np.percentile(np.concatenate([n0s, n1s]), 99.7))
bins_s = np.arange(0, max_plot + 2) - 0.5
axes[1].hist(n0s, bins=bins_s, density=True, alpha=0.6, label="ms=0")
axes[1].hist(n1s, bins=bins_s, density=True, alpha=0.6, label="ms=1")
axes[1].axvline(k_s - 0.5, linestyle="--")
axes[1].set_title("SCC readout (charge-based, thresholdable)")
axes[1].set_xlabel("Detected photons / shot")
axes[1].set_yscale("log")
axes[1].legend()

plt.tight_layout()
plt.show()
