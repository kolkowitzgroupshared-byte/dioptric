import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

### green and red calibaton 
green_coords_list =[
    [112.226, 119.47],
    [100.558, 79.848],
    [82.945, 92.343],
    ]
pixel_coords_list = green_coords_list
red_coords_list = [
    [78.784, 85.634],
    [69.613, 53.085],
    [55.07, 62.918],
    ]
M = cv2.getAffineTransform(
    np.float32(pixel_coords_list), np.float32(red_coords_list)
)
# New pixel coordinate for which we want to find the corresponding red coordinate
new_pixel_coord = np.array(
[       
    [100.295, 100.588],
    [112.204, 119.398],
    [100.614, 79.875],
    [83.01, 92.325],
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

sys.exit()

def generate_divisible_by_4(min_val, max_val, num_steps):
    step_size = (max_val - min_val) / (num_steps - 1)
    step_size = round(step_size / 4) * 4  # Ensure step size is divisible by 4

    values = [min_val + i * step_size for i in range(num_steps)]

    # Ensure values stay within bounds
    values = [x for x in values if x <= max_val]

    return values

# Example Usage
min_duration = 16
max_duration = 1000
num_steps = 51

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
