# -*- coding: utf-8 -*-
"""
DMD diffraction orders and focal-plane spots for 589 nm.

Default parameters are for a DLP6500-like square-pitch DMD:
    wavelength = 589 nm
    pixel pitch = 7.56 um

For diagonal/corner illumination, the reciprocal-lattice spacing along
the diagonal is larger by sqrt(2), so diagonal orders (q,q) obey

    sin(theta_q) = sin(theta_i) + q * sqrt(2) * lambda / p

For a 1-D row/column calculation instead, use geometry = "axis":

    sin(theta_m) = sin(theta_i) + m * lambda / p

A lens converts outgoing angle into a position in its back focal plane:

    x_m = f * tan(theta_m)

The finite illuminated beam size makes each diffraction order a finite
spot rather than a delta function.  For an approximately Gaussian beam
of 1/e^2 radius w on the DMD, a useful paraxial estimate is

    w_focus ~= lambda * f / (pi * w)

This script makes three separate figures:
    1. Diffraction angle vs incident angle
    2. Focal-plane order position vs incident angle
    3. Focal-plane spot pattern at one selected incident angle
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

WAVELENGTH_NM = 589.0
PIXEL_PITCH_UM = 7.56

# "diagonal" is appropriate for the DLP6500 corner/diagonal tilt plane.
# "axis" gives a simple row/column 1-D grating calculation.
GEOMETRY = "diagonal"

# Requested orders.  Add +/-3 or +/-4 if desired.
ORDERS = [-2, -1, 0, 1, 2]

INCIDENT_MIN_DEG = -20.0
INCIDENT_MAX_DEG = 20.0 
N_INCIDENT_POINTS = 1601

# Angle used for the single focal-plane spot plot.
CHOSEN_INCIDENT_DEG = 12.0

# Lens after/refocusing the diffracted DMD beam.
FOCAL_LENGTH_MM = 100.0

# Gaussian beam radius on DMD (1/e^2 intensity radius).
# This controls the width of each diffraction-order spot.
BEAM_RADIUS_ON_DMD_MM = 0.50

# Number of x points in simulated focal-plane intensity.
N_X = 6000


# ============================================================
# CALCULATIONS
# ============================================================

lam = WAVELENGTH_NM * 1e-9
pitch = PIXEL_PITCH_UM * 1e-6
f_m = FOCAL_LENGTH_MM * 1e-3
w_dmd_m = BEAM_RADIUS_ON_DMD_MM * 1e-3

if GEOMETRY.lower() == "diagonal":
    grating_increment = np.sqrt(2.0) * lam / pitch
    geometry_name = "diagonal (q,q) DMD orders"
elif GEOMETRY.lower() == "axis":
    grating_increment = lam / pitch
    geometry_name = "row/column 1-D DMD orders"
else:
    raise ValueError("GEOMETRY must be 'diagonal' or 'axis'")


def theta_out_deg(theta_in_deg, order):
    """
    Outgoing order angle in the unfolded/reflection-grating convention.

    sin(theta_out) = sin(theta_in) + order * grating_increment

    NaN is returned when the requested order is not propagating.
    """
    theta_in = np.deg2rad(np.asarray(theta_in_deg, dtype=float))
    s = np.sin(theta_in) + order * grating_increment

    result = np.full(np.shape(s), np.nan, dtype=float)
    valid = np.abs(s) <= 1.0
    result[valid] = np.rad2deg(np.arcsin(s[valid]))
    return result


def focal_x_mm(theta_deg):
    """Position of a plane-wave order in the back focal plane."""
    return FOCAL_LENGTH_MM * np.tan(np.deg2rad(theta_deg))


# ============================================================
# FIGURE 1: ORDER ANGLES VS INCIDENT ANGLE
# ============================================================

theta_in_scan = np.linspace(
    INCIDENT_MIN_DEG, INCIDENT_MAX_DEG, N_INCIDENT_POINTS
)

fig1, ax1 = plt.subplots(figsize=(9, 6))

for order in ORDERS:
    th = theta_out_deg(theta_in_scan, order)
    ax1.plot(theta_in_scan, th, label=f"order {order:+d}")

ax1.axvline(CHOSEN_INCIDENT_DEG, linestyle="--", linewidth=1)
ax1.set_xlabel("Incident angle relative to DMD-array normal (deg)")
ax1.set_ylabel("Outgoing diffraction angle (deg)")
ax1.set_title(
    f"DMD diffraction orders at {WAVELENGTH_NM:.0f} nm\n"
    f"{geometry_name}, pitch = {PIXEL_PITCH_UM:.2f} µm"
)
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()


# ============================================================
# FIGURE 2: ORDER SPOT POSITIONS AFTER A LENS
# ============================================================

fig2, ax2 = plt.subplots(figsize=(9, 6))

for order in ORDERS:
    th = theta_out_deg(theta_in_scan, order)
    x = focal_x_mm(th)
    ax2.plot(theta_in_scan, x, label=f"order {order:+d}")

ax2.axvline(CHOSEN_INCIDENT_DEG, linestyle="--", linewidth=1)
ax2.set_xlabel("Incident angle relative to DMD-array normal (deg)")
ax2.set_ylabel("Position in lens focal plane (mm)")
ax2.set_title(
    f"Focal-plane position of DMD orders\n"
    f"f = {FOCAL_LENGTH_MM:.1f} mm"
)
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.tight_layout()


# ============================================================
# FIGURE 3: FINITE SPOTS IN THE FOCAL PLANE
# ============================================================

# Gaussian Fourier-limited 1/e^2 radius in focal plane.
w_focus_m = lam * f_m / (np.pi * w_dmd_m)
w_focus_mm = w_focus_m * 1e3

order_data = []
for order in ORDERS:
    th = theta_out_deg(np.array([CHOSEN_INCIDENT_DEG]), order)[0]
    if np.isfinite(th):
        x0 = focal_x_mm(th)
        order_data.append((order, th, x0))

if not order_data:
    raise RuntimeError("None of the requested orders propagate.")

x_centers = np.array([item[2] for item in order_data])

padding = max(1.0, 6.0 * w_focus_mm)
x_plot = np.linspace(
    np.min(x_centers) - padding,
    np.max(x_centers) + padding,
    N_X
)

# Equal order amplitudes here only to illustrate geometry.
# Real DMD order powers are weighted by the single-micromirror blaze
# envelope, fill factor, pattern, polarization, and finite aperture.
intensity_total = np.zeros_like(x_plot)

fig3, ax3 = plt.subplots(figsize=(10, 5))

for order, th, x0 in order_data:
    # Gaussian intensity with 1/e^2 radius w_focus_mm.
    intensity = np.exp(-2.0 * ((x_plot - x0) / w_focus_mm) ** 2)
    intensity_total += intensity
    ax3.plot(x_plot, intensity, label=f"{order:+d}: {th:.2f}°")

ax3.plot(
    x_plot,
    intensity_total,
    linewidth=2.0,
    label="sum (illustrative)",
)

for order, th, x0 in order_data:
    ax3.axvline(x0, linestyle=":", linewidth=0.9)
    ax3.text(
        x0,
        1.03,
        f"{order:+d}",
        ha="center",
        va="bottom",
    )

ax3.set_xlabel("Position in lens focal plane (mm)")
ax3.set_ylabel("Relative intensity")
ax3.set_title(
    f"Where the diffraction orders form spots\n"
    f"incident = {CHOSEN_INCIDENT_DEG:.1f}°, "
    f"beam radius on DMD = {BEAM_RADIUS_ON_DMD_MM:.2f} mm"
)
ax3.grid(True, alpha=0.3)
ax3.legend()
fig3.tight_layout()


# ============================================================
# PRINT NUMERICAL VALUES
# ============================================================

print()
print("============================================================")
print("589-nm DMD diffraction calculation")
print("============================================================")
print(f"geometry              : {geometry_name}")
print(f"wavelength            : {WAVELENGTH_NM:.3f} nm")
print(f"pixel pitch           : {PIXEL_PITCH_UM:.3f} um")
print(f"incident angle        : {CHOSEN_INCIDENT_DEG:.3f} deg")
print(f"lens focal length     : {FOCAL_LENGTH_MM:.3f} mm")
print(f"beam radius on DMD    : {BEAM_RADIUS_ON_DMD_MM:.3f} mm")
print(f"estimated spot radius : {w_focus_mm*1e3:.2f} um (1/e^2)")
print()
print(" order      theta_out (deg)      x_focal (mm)")
print("------------------------------------------------------------")

for order, th, x0 in order_data:
    print(f"{order:>+5d}        {th:>10.4f}          {x0:>10.4f}")

# Relative separations from the 0th order.
zero_matches = [item for item in order_data if item[0] == 0]
if zero_matches:
    x_zero = zero_matches[0][2]
    print()
    print("Relative to the 0th-order focal spot:")
    print(" order       delta-x (mm)")
    print("--------------------------")
    for order, th, x0 in order_data:
        print(f"{order:>+5d}        {x0-x_zero:>10.4f}")

plt.show()
