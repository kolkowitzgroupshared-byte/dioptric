# from datetime import datetime

# import matplotlib.pyplot as plt
# import numpy as np


# def gaussian_phase(x, y, x0, y0, sigma, amplitude):
#     return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


# def initial_phase_pattern(shape, spots):
#     phase = np.zeros(shape, dtype=np.complex128)  # Initialize as complex array
#     for spot in spots:
#         x0, y0 = spot
#         phase += np.exp(1j * (x * x0 + y * y0))  # blaze phase
#     return np.angle(phase)


# def compute_intensity(phase):
#     return (
#         np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.exp(1j * phase))))) ** 2
#     )


# # Parameters
# shape = (256, 256)
# sigma = 200
# amplitude = 15
# spots = [(128, 128), (64, 64)]  # Example spot locations

# # Create meshgrid
# x = np.linspace(0, shape[0] - 1, shape[0])
# y = np.linspace(0, shape[1] - 1, shape[1])
# x, y = np.meshgrid(x, y)

# # Initial phase pattern
# initial_phase = initial_phase_pattern(shape, spots)

# # Gaussian phase
# gaussian_phase_profile = gaussian_phase(
#     x, y, shape[0] // 2, shape[1] // 2, sigma, amplitude
# )

# # Modified phase pattern
# modified_phase = initial_phase + gaussian_phase_profile

# # Compute intensities
# initial_intensity = compute_intensity(initial_phase)
# modified_intensity = compute_intensity(modified_phase)

# # Plotting
# plt.figure(figsize=(12, 12))

# plt.subplot(2, 2, 1)
# plt.title("Initial Phase")
# plt.imshow(initial_phase, cmap="gray")
# plt.colorbar()

# plt.subplot(2, 2, 2)
# plt.title("Initial Intensity")
# plt.imshow(initial_intensity, cmap="hot")
# plt.colorbar()

# plt.subplot(2, 2, 3)
# plt.title("Gaussian Phase Profile")
# plt.imshow(gaussian_phase_profile, cmap="gray")
# plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.title("Modified Intensity")
# plt.imshow(modified_intensity, cmap="hot")
# plt.colorbar()

# plt.tight_layout()
# # Save figure with current date and time
# current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
# plt.savefig(save_path)
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Define grid
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)

# # Constants
# x0 = 5
# y0 = 5

# # Complex exponential function
# z = np.exp(1j * (x * x0 + y * y0))

# # Plot real part
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(np.real(z), extent=(-5, 5, -5, 5), cmap="viridis")
# plt.colorbar(label="Real part")
# plt.title("Real part of exp(1j * (x * x0 + y * y0))")

# # Plot imaginary part
# plt.subplot(1, 2, 2)
# plt.imshow(np.imag(z), extent=(-5, 5, -5, 5), cmap="viridis")
# plt.colorbar(label="Imaginary part")
# plt.title("Imaginary part of exp(1j * (x * x0 + y * y0))")

# plt.tight_layout()
# # Save figure with current date and time
# # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# # save_path = f"G:/My Drive/Experiments/SLM_seup_data/image_{current_datetime}.png"
# # plt.savefig(save_path)
# plt.show()
import sys

# import numpy as np
# # Step 1: Convert to numpy arrays
# green_old = np.array([[118.127, 97.472], [107.036, 118.416], [96.822, 94.821]])
# red_old = np.array([[80.703, 64.786], [72.119, 81.942], [63.276, 62.851]])
# green_new = np.array([[120.667, 95.464], [104.513, 119.016], [96.313, 92.987]])
# # Step 2: Add ones for affine transform (homogeneous coords)
# G = np.hstack([green_old, np.ones((3, 1))])  # 3x3
# R = red_old  # 3x2
# # Solve for affine matrix: G @ M = R → M = (G^T G)^(-1) G^T R
# M, _, _, _ = np.linalg.lstsq(G, R, rcond=None)  # M is 3x2
# # Step 3: Apply same transform to new green coordinates
# G_new = np.hstack([green_new, np.ones((green_new.shape[0], 1))])  # Nx3
# red_new = G_new @ M  # Nx2
# print("Estimated new red coordinates:")
# print(red_new)
# sys.exit()
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
### green and red calibaton at RT setup 2025-09-15
pixel_coords_list = [[119.522, 118.997], [111.538, 95.186], [96.194, 118.343]]
red_coords_list = [[82.395, 81.819], [76.707, 62.056], [63.349, 80.092]]
# Given pixel coordinates and corresponding red coordinates
# pixel_coords_list = np.array(
#     [
#         [122.217, 28.629],
#         [27.396, 150.264],
#         [220.452, 192.19],
#     ]
# )
# red_coords_list = np.array(
#     [
#         [72.293, 80.694],
#         [82.931, 71.034],
#         [65.743, 64.346],
#     ]
# )

## greeen
# pixel_coords_list = np.array([[27.805, 30.303], [138.235, 235.17], [228.671, 15.343]])
# red_coords_list = np.array([[[29.437, 41.77], [147.414, 233.083], [224.636, 21.373]]])

# For two points, a simpler method is necessary, but let's try using cv2.estimateAffinePartial2D
if len(pixel_coords_list) >= 3:
    # Use cv2.estimateAffinePartial2D to get the affine transformation matrix
    M = cv2.getAffineTransform(
        np.float32(pixel_coords_list), np.float32(red_coords_list)
    )

    # New pixel coordinate for which we want to find the corresponding red coordinate
    new_pixel_coord = np.array(
        [
        [107.884, 107.983],
        [119.262, 119.511],
        [111.272, 95.718],
        [95.966, 118.875],
        ],
        dtype=np.float32,
    )

    # Apply the affine transformation to the new pixel coordinate
    new_red_coord = cv2.transform(np.array([new_pixel_coord]), M)

    # Print the corresponding red coordinates
    print("[")
    for coord in new_red_coord[0]:
        rounded_coord = [round(x, 3) for x in coord]
        print(f"    {rounded_coord},")
    print("]")
else:
    # Calculate manually if only two points are available
    # Define the simple transformation function
    def simple_transform(pixel_point, src_points, dst_points):
        # Calculate scaling and translation manually
        scale_x = (dst_points[1][0] - dst_points[0][0]) / (
            src_points[1][0] - src_points[0][0]
        )
        scale_y = (dst_points[1][1] - dst_points[0][1]) / (
            src_points[1][1] - src_points[0][1]
        )

        # Calculate translation
        translation_x = dst_points[0][0] - scale_x * src_points[0][0]
        translation_y = dst_points[0][1] - scale_y * src_points[0][1]

        # Apply transformation
        new_x = scale_x * pixel_point[0] + translation_x
        new_y = scale_y * pixel_point[1] + translation_y

        return np.array([new_x, new_y])

    # New pixel coordinates to transform
    new_pixel_coord = np.array(
        [
            [108.368, 107.114],
            [119.483, 121.535],
            [106.762, 93.524],
            [93.748, 118.277],
        ],
        dtype=np.float32,
    )

    # Apply the transformation to each new pixel coordinate
    transformed_red_coords = [
        simple_transform(coord, pixel_coords_list, red_coords_list)
        for coord in new_pixel_coord
    ]

    # Print the transformed red coordinates
    print("[")
    for coord in transformed_red_coords:
        rounded_coord = [round(x, 3) for x in coord]
        print(f"    {rounded_coord},")
    print("]")

    # Calculate using simple linear transform
    # new_red_coord = simple_transform(
    #     [42.749, 125.763], pixel_coords_list, red_coords_list
    # )
    # print("Corresponding red coordinates:", new_red_coord)
# sys.exit()

min_tau = 200  # ns
max_tau = 100e3  # fallback if no revival_period given
taus = []

# Densely sample early decay
decay_width = 5e3
decay = np.linspace(min_tau, min_tau + decay_width, 6)
taus.extend(decay.tolist())

taus.extend(np.geomspace(min_tau + decay_width, max_tau, 81).tolist())

# Round to clock-cycle-compatible units
taus = [round(el / 4) * 4 for el in taus]

# Remove duplicates and sort
taus = sorted(set(taus))
taus_x = np.linspace(1, len(taus), len(taus))
plt.figure()
plt.scatter(taus_x, taus)
plt.show(block=True)
# sys.exit()


def generate_divisible_by_4(min_val, max_val, num_steps):
    step_size = (max_val - min_val) / (num_steps - 1)
    step_size = round(step_size / 4) * 4  # Ensure step size is divisible by 4

    values = [min_val + i * step_size for i in range(num_steps)]

    # Ensure values stay within bounds
    values = [x for x in values if x <= max_val]

    return values


# Example Usage
min_duration = 16
max_duration = 210
num_steps = 18

step_values = generate_divisible_by_4(min_duration, max_duration, num_steps)
print(step_values)
print(len(step_values))
sys.exit()
def logspace_div4(min_val, max_val, num_steps, base=10.0):
    if not (min_val > 0 and max_val > min_val and num_steps >= 2):
        raise ValueError("Bad inputs.")
    m_lo, m_hi = int(np.ceil(min_val / 4)), int(np.floor(max_val / 4))
    if m_lo > m_hi:
        raise ValueError("Range too narrow for multiples of 4.")
    k = 8
    while True:
        xs = np.logspace(
            np.log(m_lo) / np.log(base),
            np.log(m_hi) / np.log(base),
            k * num_steps,
            base=base,
        )
        m = np.unique(np.clip(np.rint(xs).astype(int), m_lo, m_hi))
        if m[0] != m_lo:
            m[0] = m_lo
        if m[-1] != m_hi:
            m[-1] = m_hi
        if len(m) >= num_steps:
            break
        k *= 2
    idx = np.linspace(0, len(m) - 1, num_steps).round().astype(int)
    return (m[idx] * 4).tolist()


# Example
print(logspace_div4(16, 200, 18))
sys.exit()

# sys.exit()
# Updating plot with center frequencies in the legend
# # Given data
# green_aod_freq_MHz = np.array([90, 95, 100, 105, 110, 115, 120, 125])
# green_laser_power_uW = np.array([260, 310, 330, 350, 360, 340, 240, 140])
# red_aod_freq_MHz = np.array([55, 60, 65, 70, 75, 80, 85, 90])
# red_laser_power_uW = np.array([112, 200, 255, 260, 270, 260, 205, 110])
# # Define center frequencies and compute x-axis difference
# green_center_freq = 110  # MHz
# red_center_freq = 75  # MHz
# green_x_diff = green_aod_freq_MHz - green_center_freq
# red_x_diff = red_aod_freq_MHz - red_center_freq
# # Normalize the laser powers using 0 uW as the minimum
# green_laser_power_normalized = green_laser_power_uW / green_laser_power_uW.max()
# red_laser_power_normalized = red_laser_power_uW / red_laser_power_uW.max()
# # Plotting
# plt.figure(figsize=(7, 5))
# plt.plot(
#     green_x_diff,
#     green_laser_power_normalized,
#     label="Green Laser Power (Center: 110 MHz)",
#     marker="o",
# )
# plt.plot(
#     red_x_diff,
#     red_laser_power_normalized,
#     label="Red Laser Power (Center: 75 MHz)",
#     marker="s",
# )
# plt.xlabel("Frequency Difference from Center (MHz)")
# plt.ylabel("Normalized Laser Power (uW)")
# plt.title("Normalized Laser Power vs Frequency Difference from Center")
# plt.legend()
# plt.grid(True)
# plt.show()
# Try a logarithmic function instead, which might better capture the relationship
from utils import kplotlib as kpl

kpl.init_kplotlib()

aom_voltages = np.array(
    [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35], dtype=float
)
yellow_power = np.array(
    [34, 54, 81, 118, 162, 214, 272, 334, 404, 490, 585], dtype=float
)


def power_law_model(x, a, b, c):
    return a * x**b + c


# --- Step 1: log–log fit (no offset) to seed a, b
# Guard against zeros/negatives
mask = (aom_voltages > 0) & (yellow_power > 0)
x_pos = aom_voltages[mask]
y_pos = yellow_power[mask]

B = np.polyfit(np.log(x_pos), np.log(y_pos), 1)  # log(y)= b*log(x) + log(a)
b0 = B[0]
a0 = np.exp(B[1])
c0 = 0.0  # start with no offset; curve_fit will adjust

p0 = [a0, b0, c0]

# Optional: constrain c to be >= 0 (or near min power); keeps optimizer sane
lower_bounds = (
    0.0,
    -10.0,
    -0.2 * yellow_power.max(),
)  # allow small negative c if you want
upper_bounds = (np.inf, 10.0, 0.5 * yellow_power.max())

# --- Step 2: fit with better initials + more iterations
params, covariance = curve_fit(
    power_law_model,
    aom_voltages,
    yellow_power,
    p0=p0,
    bounds=(lower_bounds, upper_bounds),
    maxfev=100000,
)
a, b, c = params

# --- Generate smooth curve
voltage_fit = np.linspace(aom_voltages.min(), aom_voltages.max(), 500)
power_fit = power_law_model(voltage_fit, a, b, c)
print(f"Fitted power-law: y = {a:.6g} * x^{b:.6g} + {c:.6g}")

# --- Plot in linear scale
plt.figure(figsize=(7, 5))
plt.scatter(aom_voltages, yellow_power, label="Data")
plt.plot(
    voltage_fit, power_fit, label=f"Power-law fit: y = {a:.3g}·x^{b:.3g} + {c:.3g}"
)
plt.title("AOM Voltage vs Yellow Laser Power")
plt.xlabel("AOM Voltage (V)")
plt.ylabel("Yellow Laser Power (µW)")
plt.grid(True, alpha=0.3)
plt.legend()
# plt.tight_layout()


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, curve_fit

# fmt: off
# -----------------------------
# Data 2025_09_14
# -----------------------------
green_x = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], dtype=float)
green_y = np.array([11, 147, 683, 1960, 4220, 7330, 10740], dtype=float)

red_x = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], dtype=float)
red_y = np.array([28, 356, 1490, 4320, 9400, 16900, 26000], dtype=float)


yellow_x = np.array(
    [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35], dtype=float
)
yellow_y = np.array([34, 54, 81, 118, 162, 214, 272, 334, 404, 490, 585], dtype=float)

# -----------------------------
# Data 2025_09_17
# -----------------------------
green_x = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], dtype=float)
green_y = np.array([12, 162, 752, 2170, 4520, 7660, 11500, 15400], dtype=float)

red_x = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16], dtype=float)
red_y = np.array([24, 303, 1430, 4160, 9000, 16200, 25000, 34200], dtype=float)


yellow_x = np.array(
    [0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39], dtype=float
)
yellow_y = np.array([15, 29, 51, 81, 121, 175, 242, 322, 410, 504, 602, 725, 865, 1002, 1140], dtype=float)

# fmt: on


# -----------------------------
# Model and utilities
# -----------------------------
def power_law_model(x, a, b, c):
    """y = a * x^b + c"""
    return a * np.power(x, b) + c


def seed_from_loglog(x, y):
    """Initial guess (a, b) from log–log fit, ignore offset c."""
    m = (x > 0) & (y > 0)
    B = np.polyfit(np.log(x[m]), np.log(y[m]), 1)  # log y = b log x + log a
    b0 = float(B[0])
    a0 = float(np.exp(B[1]))
    return a0, b0


def fit_power_law_with_offset(x, y, allow_negative_c_frac=0.2, c_upper_frac=0.5):
    a0, b0 = seed_from_loglog(x, y)
    c0 = 0.0
    p0 = [a0, b0, c0]

    lb = (0.0, -10.0, -allow_negative_c_frac * np.max(y))
    ub = (np.inf, 10.0, c_upper_frac * np.max(y))

    params, cov = curve_fit(
        power_law_model, x, y, p0=p0, bounds=(lb, ub), maxfev=100000
    )
    return params, cov


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def predict_power(x, params):
    """Predict power for amplitude x."""
    a, b, c = params
    return power_law_model(np.asarray(x, float), a, b, c)


def invert_amp_for_power(target_power, params, x_min, x_max):
    """Solve for amplitude x that gives target_power within [x_min, x_max]."""
    a, b, c = params
    f = lambda x: power_law_model(x, a, b, c) - target_power
    # Ensure the bracket contains a root; expand a bit if needed
    xa, xb = float(x_min), float(x_max)
    fa, fb = f(xa), f(xb)
    if fa * fb > 0:
        # try expanding slightly
        xa = max(1e-6, 0.9 * x_min)
        xb = 1.1 * x_max
        fa, fb = f(xa), f(xb)
        if fa * fb > 0:
            raise ValueError(
                "Bracket does not contain a root for the given target power."
            )
    return brentq(f, xa, xb)


def make_plots(label, x, y, params):
    a, b, c = params
    x_fit = np.linspace(min(x), max(x), 500)
    y_fit = power_law_model(x_fit, a, b, c)

    # Linear plot
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label=f"{label} data")
    plt.plot(x_fit, y_fit, label=f"Fit: y = {a:.3g}·x^{b:.3g} + {c:.3g}")
    plt.title(f"{label}: AOD Amplitude vs Laser Power")
    plt.xlabel("AOD Amplitude (arb. V)")
    plt.ylabel("Laser Power (µW)")
    plt.grid(True, alpha=0.3)
    plt.legend()


# -----------------------------
# Fit and report
# -----------------------------
green_params, green_cov = fit_power_law_with_offset(green_x, green_y)
red_params, red_cov = fit_power_law_with_offset(red_x, red_y)
yellow_params, yellow_cov = fit_power_law_with_offset(yellow_x, yellow_y)

print("Green params [a, b, c]:", green_params)
print("Red   params [a, b, c]:", red_params)
print("Yellow   params [a, b, c]:", yellow_params)

print("Green R^2:", r2_score(green_y, predict_power(green_x, green_params)))
print("Red   R^2:", r2_score(red_y, predict_power(red_x, red_params)))
print("Yellow  R^2:", r2_score(yellow_y, predict_power(yellow_x, red_params)))

# -----------------------------
# Plots
# -----------------------------
make_plots("Green", green_x, green_y, green_params)
make_plots("Red", red_x, red_y, red_params)
make_plots("Yellow", yellow_x, yellow_y, yellow_params)

# -----------------------------
# Examples: prediction and inversion
# -----------------------------
# Predict power at x = 0.11
x_query = 0.22
print("Green power at x=0.11:", float(predict_power(x_query, green_params)))
print("Red   power at x=0.11:", float(predict_power(x_query, red_params)))
print("Yellow  power at x=0.11:", float(predict_power(x_query, yellow_params)))

# Invert: which amplitude gives 10 mW (10000 µW)?
target_power = 184.0
gx = invert_amp_for_power(target_power, green_params, green_x.min(), green_x.max())
rx = invert_amp_for_power(target_power, red_params, red_x.min(), red_x.max())
yx = invert_amp_for_power(target_power, yellow_params, yellow_x.min(), yellow_x.max())
print(f"Green amplitude for {target_power:.0f} µW:", round(gx, 4))
print(f"Red   amplitude for {target_power:.0f} µW:", round(rx, 4))
print(f"Yellow   amplitude for {target_power:.0f} µW:", round(yx))


plt.show(block=True)
