import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
file_path = r"C:\Users\Saroj Chand\Downloads\Davis_118 Trend.csv"
df = pd.read_csv(file_path)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

temp_col = "Lab118SpaceTemp"
y = df[temp_col].to_numpy(dtype=float)

# Sampling interval
tau0 = df["Timestamp"].diff().dt.total_seconds().dropna().median()
dt_minutes = tau0 / 60
print(f"Sampling interval = {dt_minutes:.1f} min")

# 1-hour rolling window for ~15 min data
rolling_window = max(2, int(round(60 / dt_minutes)))
print(f"Rolling window (~1 hour) = {rolling_window} points")

# =========================
# Basic stats
# =========================
mean_temp = df[temp_col].mean()
std_temp = df[temp_col].std()
var_temp = df[temp_col].var()
min_temp = df[temp_col].min()
max_temp = df[temp_col].max()
temp_range = max_temp - min_temp
cv_percent = 100 * std_temp / mean_temp

x_hours = (df["Timestamp"] - df["Timestamp"].min()).dt.total_seconds() / 3600
slope_f_per_hour, intercept = np.polyfit(x_hours, y, 1)
trend_line = intercept + slope_f_per_hour * x_hours
slope_f_per_day = slope_f_per_hour * 24

print("\nOverall statistics")
print(f"Mean temperature      : {mean_temp:.4f} °F")
print(f"Standard deviation    : {std_temp:.4f} °F")
print(f"Variance              : {var_temp:.6f} °F²")
print(f"Min / Max             : {min_temp:.4f} / {max_temp:.4f} °F")
print(f"Range                 : {temp_range:.4f} °F")
print(f"Coeff. of variation   : {cv_percent:.3f} %")
print(f"Drift slope           : {slope_f_per_hour:.5f} °F/hour")
print(f"Drift slope           : {slope_f_per_day:.5f} °F/day")

# Rolling stats
df["Temp_rolling_mean_1h"] = (
    df[temp_col].rolling(window=rolling_window, center=True).mean()
)
df["Temp_rolling_std_1h"] = (
    df[temp_col].rolling(window=rolling_window, center=True).std()
)
df["Temp_rolling_var_1h"] = (
    df[temp_col].rolling(window=rolling_window, center=True).var()
)


# =========================
# Allan / Hadamard deviation
# =========================
def linear_detrend(x):
    n = len(x)
    t = np.arange(n, dtype=float)
    m, b = np.polyfit(t, x, 1)
    return x - (m * t + b)


def allan_deviation(x, tau0, max_m=None):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if max_m is None:
        max_m = max(1, n // 10)

    m_values = []
    adev_values = []

    m = 1
    while m <= max_m:
        k = n // m
        if k < 3:
            break

        x_trim = x[: k * m]
        x_avg = x_trim.reshape(k, m).mean(axis=1)

        diffs = np.diff(x_avg)
        avar = 0.5 * np.mean(diffs**2)
        adev = np.sqrt(avar)

        m_values.append(m)
        adev_values.append(adev)
        m *= 2

    taus = tau0 * np.array(m_values)
    adevs = np.array(adev_values)
    return taus, adevs


def hadamard_deviation(x, tau0, max_m=None):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if max_m is None:
        max_m = max(1, n // 10)

    m_values = []
    hdev_values = []

    m = 1
    while m <= max_m:
        k = n // m
        if k < 4:
            break

        x_trim = x[: k * m]
        x_avg = x_trim.reshape(k, m).mean(axis=1)

        d2 = x_avg[2:] - 2 * x_avg[1:-1] + x_avg[:-2]
        hvar = np.mean(d2**2) / 6.0
        hdev = np.sqrt(hvar)

        m_values.append(m)
        hdev_values.append(hdev)
        m *= 2

    taus = tau0 * np.array(m_values)
    hdevs = np.array(hdev_values)
    return taus, hdevs


y_detrended = linear_detrend(y)

taus_raw, adev_raw = allan_deviation(y, tau0)
taus_det, adev_det = allan_deviation(y_detrended, tau0)
taus_h, hdev = hadamard_deviation(y, tau0)

# =========================
# Daily summary
# =========================
daily_stats = (
    df.set_index("Timestamp")[temp_col]
    .resample("D")
    .agg(["mean", "std", "var", "min", "max"])
)
daily_stats["range"] = daily_stats["max"] - daily_stats["min"]

print("\nDaily statistics")
print(daily_stats)

# =========================
# Hour-of-day summary
# =========================
df["HourFloat"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60

hourly_stats = (
    df.groupby("HourFloat")[temp_col].agg(["mean", "std", "count"]).reset_index()
)

# =========================
# Stability assessment
# =========================
if std_temp < 0.3 and abs(slope_f_per_day) < 0.2:
    stability_msg = "Overall fairly stable; only a small slow drift."
elif std_temp < 0.5 and abs(slope_f_per_day) < 0.5:
    stability_msg = (
        "Moderately stable, with noticeable but not large drift/fluctuation."
    )
else:
    stability_msg = "Not especially stable; fluctuations and/or drift are significant."

print("\nStability assessment:")
print(stability_msg)

# =========================
# Plot 1: Raw + rolling mean + trend
# =========================
plt.figure(figsize=(12, 5))
plt.plot(df["Timestamp"], df[temp_col], ".", alpha=0.6, label="Raw")
plt.plot(
    df["Timestamp"],
    df["Temp_rolling_mean_1h"],
    "-",
    linewidth=2,
    label="1-hour rolling mean",
)
plt.plot(
    df["Timestamp"],
    trend_line,
    "--",
    linewidth=2,
    label=f"Linear trend ({slope_f_per_day:.3f} °F/day)",
)
plt.xlabel("Time")
plt.ylabel("Temperature (°F)")
plt.title("Lab 118 Space Temperature")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Plot 2: Rolling standard deviation
# =========================
plt.figure(figsize=(12, 4.5))
plt.plot(
    df["Timestamp"],
    df["Temp_rolling_std_1h"],
    "-",
    linewidth=2,
    label="1-hour rolling std",
)
plt.axhline(std_temp, linestyle="--", label=f"Overall std = {std_temp:.3f} °F")
plt.xlabel("Time")
plt.ylabel("Std (°F)")
plt.title("Rolling Standard Deviation")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Plot 3: Rolling variance
# =========================
plt.figure(figsize=(12, 4.5))
plt.plot(
    df["Timestamp"],
    df["Temp_rolling_var_1h"],
    "-",
    linewidth=2,
    label="1-hour rolling variance",
)
plt.axhline(var_temp, linestyle="--", label=f"Overall var = {var_temp:.4f} °F²")
plt.xlabel("Time")
plt.ylabel("Variance (°F²)")
plt.title("Rolling Variance")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Plot 4: Histogram
# =========================
plt.figure(figsize=(7, 4.5))
plt.hist(df[temp_col], bins=25, edgecolor="black")
plt.xlabel("Temperature (°F)")
plt.ylabel("Count")
plt.title("Temperature Distribution")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# =========================
# Plot 5: Daily range
# =========================
plt.figure(figsize=(8, 4.5))
plt.bar(daily_stats.index.astype(str), daily_stats["range"])
plt.xlabel("Date")
plt.ylabel("Daily range (°F)")
plt.title("Daily Temperature Range")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# =========================
# Plot 6: Average temperature vs hour of day
# =========================
plt.figure(figsize=(9, 4.5))
plt.errorbar(
    hourly_stats["HourFloat"],
    hourly_stats["mean"],
    yerr=hourly_stats["std"],
    fmt="o-",
    capsize=3,
)
plt.xlabel("Hour of day")
plt.ylabel("Temperature (°F)")
plt.title("Average Temperature vs Hour of Day")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# =========================
# Plot 7: Allan / Hadamard deviation
# =========================
plt.figure(figsize=(8, 5.5))
plt.loglog(taus_raw / 3600, adev_raw, "o-", label="Allan deviation (raw)")
plt.loglog(taus_det / 3600, adev_det, "s-", label="Allan deviation (detrended)")
plt.loglog(taus_h / 3600, hdev, "^-", label="Hadamard deviation")
plt.xlabel("Averaging time τ (hours)")
plt.ylabel("Deviation (°F)")
plt.title("Temperature Stability Analysis")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
