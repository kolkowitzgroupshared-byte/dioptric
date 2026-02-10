# -*- coding: utf-8 -*-
"""
Created on June 16th, 2023
@author: Saroj B Chand
"""
import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy import stats  # for linear regression drift estimate
from utils import kplotlib as kplt

kplt.init_kplotlib()
# ----------------------------
# User-configurable parameters
# ----------------------------
base_folder = "G:\\NV_Widefield_RT_Setup_Enclosure_Temp_Logs"
hours = 4  # window to analyze & plot
temp_low, temp_high = 15, 35  # sanity filter limits
PLOT_ADEV = True  # set False if you don't want the Allan plot refreshing
# Define channels and corresponding filenames
channels = {
    "4A": "temp_4A.csv",
    "4B": "temp_4B.csv",
    "4C": "temp_4C.csv",
    "4D": "temp_4D.csv",
    "temp_stick": "temp_stick.csv",
}

# ----------------------------
# Dynamic Allan taus
# ----------------------------
def build_allan_taus(hours: float) -> list:
    # From 60 s (1 min) to the full window, at least 8 points
    n_pts = max(8, int(hours * 2))  # ~2 points/hour
    return list(np.linspace(60, hours * 3600, n_pts))


ALLAN_TAUS = build_allan_taus(hours)
# ----------------------------
# Folder bookkeeping
# ----------------------------
now = datetime.datetime.now()
folder_current = now.strftime("%m%Y")
folder_previous = (now - relativedelta(months=1)).strftime("%m%Y")

data_folders = [
    os.path.join(base_folder, folder_previous),
    os.path.join(base_folder, folder_current),
]
# ----------------------------
# Plot setup
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
fig_adev, ax_adev = plt.subplots(figsize=(7, 5)) if PLOT_ADEV else (None, None)

timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Current Time Stamp: {timestamp}")


# ----------------------------
# Helpers
# ----------------------------
def load_channel_df(filename: str) -> pd.DataFrame:
    """Load and concatenate current+previous month CSVs for a channel."""
    dfs = []
    for folder in data_folders:
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            continue
        try:
            df = pd.read_csv(
                file_path,
                names=["Timestamp", "Temperature"],
                parse_dates=["Timestamp"],
                dtype={"Temperature": float},
            )
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not dfs:
        return pd.DataFrame(columns=["Timestamp", "Temperature"])

    return pd.concat(dfs, ignore_index=True)


def window_and_filter(
    df: pd.DataFrame, hours: float, t_low: float, t_high: float
) -> pd.DataFrame:
    """Apply time window and temp sanity filter."""
    if df.empty:
        return df
    now = datetime.datetime.now()
    df = df[
        (df["Timestamp"] > (now - datetime.timedelta(hours=hours)))
        & (df["Temperature"] > t_low)
        & (df["Temperature"] < t_high)
    ].copy()
    return df


def compute_drift_deg_per_hour(df: pd.DataFrame) -> float:
    """Linear fit (Temperature vs time) slope converted to °C/hour."""
    if len(df) < 3:
        return np.nan
    t = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds().values
    y = df["Temperature"].values
    slope, _, _, _, _ = stats.linregress(t, y)
    return slope * 3600.0  # °C/sec -> °C/hour


def simple_allan_deviation(
    series: pd.Series, dt_seconds: float, taus_seconds: list[int]
) -> dict:
    """
    Simple (non-overlapping) Allan deviation.
    Assumes uniformly-sampled data at dt_seconds.
    """
    y = series.values
    N = len(y)
    adev = {}
    for tau in taus_seconds:
        m = int(round(tau / dt_seconds))  # samples per tau
        if m < 1 or m > 2 * N:
            adev[tau] = np.nan
            continue
        nblocks = N // m
        y_trim = y[: nblocks * m]
        block_means = y_trim.reshape(nblocks, m).mean(axis=1)
        diff = np.diff(block_means)
        avar = 0.5 * np.mean(diff**2)
        adev[tau] = np.sqrt(avar)
    return adev


def compute_metrics(df: pd.DataFrame, allan_taus=ALLAN_TAUS) -> dict:
    """Compute std, sem, peak-to-peak, drift, Allan dev."""
    out = {
        "N": len(df),
        "mean": np.nan,
        "std": np.nan,
        "sem": np.nan,
        "min": np.nan,
        "max": np.nan,
        "p2p": np.nan,
        "drift_deg_per_hr": np.nan,
        "allan_dev": {},
        "med_dt": np.nan,
    }
    if df.empty:
        return out

    temps = df["Temperature"].values
    out["mean"] = float(np.mean(temps))
    out["std"] = float(np.std(temps, ddof=1) if len(temps) > 1 else 0.0)
    out["sem"] = float(out["std"] / np.sqrt(out["N"])) if out["N"] > 0 else np.nan
    out["min"] = float(np.min(temps))
    out["max"] = float(np.max(temps))
    out["p2p"] = float(out["max"] - out["min"])
    out["drift_deg_per_hr"] = compute_drift_deg_per_hour(df)

    # Allan deviation: resample to a uniform cadence first
    try:
        df_u = df.set_index("Timestamp").sort_index()
        med_dt = df_u.index.to_series().diff().median().total_seconds()
        out["med_dt"] = med_dt
        if pd.isna(med_dt) or med_dt <= 0:
            out["allan_dev"] = {tau: np.nan for tau in allan_taus}
        else:
            rule = f"{int(round(med_dt))}S"
            df_res = df_u["Temperature"].resample(rule).mean().interpolate("time")
            out["allan_dev"] = simple_allan_deviation(df_res, med_dt, allan_taus)
    except Exception as e:
        print("Allan deviation computation failed:", e)
        out["allan_dev"] = {tau: np.nan for tau in allan_taus}

    return out


def pretty_print_metrics(label: str, m: dict):
    print(f"\n=== Channel {label} ===")
    print(f"N = {m['N']}")
    print(f"mean = {m['mean']:.6f} °C")
    print(f"std  = {m['std']:.6f} °C")
    print(f"sem  = {m['sem']:.6f} °C")
    print(f"min/max = {m['min']:.6f}/{m['max']:.6f} °C")
    print(f"peak-to-peak = {m['p2p']:.6f} °C")
    print(f"drift = {m['drift_deg_per_hr']:.6e} °C/hour")
    # if m["allan_dev"]:
    #     print("Allan deviation (°C):")
    #     for tau, val in m["allan_dev"].items():
    #         if np.isnan(val):
    #             print(f"  tau = {tau:>6.0f} s :  NaN")
    #         else:
    #             print(f"  tau = {tau:>6.0f} s :  {val:.6e}")


# at top of file

LOCAL_TZ = "America/Los_Angeles"


def _coerce_temperature(series: pd.Series) -> pd.Series:
    # handles floats or strings like "21.730 °C"
    s = series.astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(s, errors="coerce")


def _normalize_timestamps(df: pd.DataFrame, col: str = "Timestamp") -> pd.DataFrame:
    # Convert anything to timezone-naive local datetimes for easy comparisons
    if np.issubdtype(df[col].dtype, np.datetime64):
        s = df[col]
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    else:
        # Parse strings/objects; assume input may be UTC and convert to local
        s = pd.to_datetime(df[col], errors="coerce")
        s = s.dt.tz_localize(None)

    df[col] = s
    df.dropna(subset=[col], inplace=True)
    return df


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" in df.columns:
        df = _normalize_timestamps(df, "Timestamp")
    if "Temperature" in df.columns:
        df["Temperature"] = _coerce_temperature(df["Temperature"])
        df.dropna(subset=["Temperature"], inplace=True)
    return df


# cache a single figure annotation so we don't add it repeatedly
_fig_note = None


def update_plot():
    global _fig_note

    # Clear axes, not the whole figure.
    ax.clear()
    if PLOT_ADEV:
        ax_adev.clear()

    for label, filename in channels.items():
        try:
            df_all = load_channel_df(filename)
            if df_all.empty:
                print(f"No data found for Channel {label}")
                continue

            # Be defensive: normalize once more here in case the loader changes later
            df_all = normalize_df(df_all)

            df_all = window_and_filter(df_all, hours, temp_low, temp_high)
            if df_all.empty:
                print(f"No recent/valid data for Channel {label}")
                continue

            # Compute & print metrics
            metrics = compute_metrics(df_all, ALLAN_TAUS)
            pretty_print_metrics(label, metrics)

            # Plot temperature (ensure sorted by time)
            df_all = df_all.sort_values("Timestamp")
            ax.plot(
                df_all["Timestamp"].to_numpy(),
                df_all["Temperature"].to_numpy(),
                label=f"{label}",
            )

            # Plot Allan deviation if available
            if PLOT_ADEV and metrics.get("allan_dev"):
                # ensure numeric + sorted taus, positive values only for loglog
                taus = np.array(
                    [float(t) for t in metrics["allan_dev"].keys()], dtype=float
                )
                vals = np.array(
                    [metrics["allan_dev"][t] for t in metrics["allan_dev"].keys()],
                    dtype=float,
                )
                # filter out NaNs and nonpositive
                good = np.isfinite(taus) & np.isfinite(vals) & (taus > 0) & (vals > 0)
                if np.any(good):
                    order = np.argsort(taus[good])
                    ax_adev.loglog(
                        taus[good][order], vals[good][order], marker="o", label=label
                    )
        except Exception as e:
            # Don't let one bad channel kill the loop
            print(f"[WARN] Skipping channel {label} due to error: {e}")

    # ----- Decorate time plot -----
    ax.set_title(f"Temperature (Last {hours} h)", fontsize=13)
    ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Temperature [°C]", fontsize=13)

    locator = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%Y-%m-%d-%H-%M-%S")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)
    fig.autofmt_xdate()

    # Single persistent note on the figure (don’t add duplicates every refresh)
    note = (
        "4A - near sample \n"
        "4B - experiment enclosure (feedback)\n"
        "4C - laser enclosure (feedback)\n"
        "4D - laser enclosure\n"
        "temp_stick - outside monitor"
    )
    if _fig_note is None:
        _fig_note = fig.text(0.70, 0.56, note, ha="left", va="bottom", fontsize=11)
    else:
        _fig_note.set_text(note)

    # ----- Decorate ADEV plot -----
    if PLOT_ADEV:
        ax_adev.set_title(f"Allan Deviation (Last {hours} h)", fontsize=13)
        ax_adev.set_xlabel("Averaging Time τ [s]", fontsize=13)
        ax_adev.set_ylabel("Allan Deviation [°C]", fontsize=13)
        ax_adev.grid(True, which="both", ls="--", alpha=0.4)
        ax_adev.legend(fontsize=11)

    plt.pause(0.1)


def main():
    print(f"Live plotting from: {data_folders}")
    try:
        while True:
            update_plot()
            # Optional: comment out the input() for fully hands-free refresh
            cmd = input("Press Enter to refresh or type 'q' to quit: ").strip().lower()
            if cmd == "q":
                break
    finally:
        print("Exiting and closing plots.")
        plt.close("all")


if __name__ == "__main__":
    main()

# from scipy.optimize import curve_fit


# def exp_step_with_drift(t, T_inf, T0, t0, tau, m):
#     t = np.asarray(t, float)
#     tt = np.maximum(0.0, t - t0)
#     return T_inf - (T_inf - T0) * np.exp(-tt / np.maximum(1e-9, tau)) + m * tt


# def end_slope_deg_per_min(dfw, tail_minutes=10):
#     """Estimate end-of-window slope (°C/min) using a robust linear fit over the last tail_minutes."""
#     t = dfw["Timestamp"].to_numpy()
#     y = dfw["Temperature"].to_numpy()
#     tsec = (t - t[0]).astype("timedelta64[s]").astype(float)

#     t_end = tsec[-1]
#     mask = tsec >= (t_end - tail_minutes * 60.0)
#     if mask.sum() < 5:
#         return np.nan

#     ts = tsec[mask]
#     ys = y[mask]
#     # simple least-squares slope in °C/s
#     A = np.vstack([ts, np.ones_like(ts)]).T
#     slope_s, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
#     return slope_s * 60.0  # °C/min


# def fit_time_constant_with_drift(df_win, label="4D", do_plot=True):
#     dfw = df_win.sort_values("Timestamp").copy()
#     tsec = (dfw["Timestamp"] - dfw["Timestamp"].iloc[0]).dt.total_seconds().to_numpy()
#     y = dfw["Temperature"].to_numpy()

#     # Initial guesses (reuse from simple model)
#     T0_guess = float(np.median(y[: max(3, len(y) // 20)]))
#     Tinf_guess = float(np.median(y[-max(3, len(y) // 20) :]))
#     t0_guess = 0.0
#     tau_guess = 30 * 60.0
#     m_guess = 0.0  # assume small drift

#     p0 = [Tinf_guess, T0_guess, t0_guess, tau_guess, m_guess]
#     bounds = (
#         [min(y) - 10, min(y) - 10, -60.0, 1.0, -0.05],  # m lower ≈ -0.05 °C/s
#         [max(y) + 10, max(y) + 10, 600.0, 6 * 3600.0, 0.05],  # m upper ≈ +0.05 °C/s
#     )

#     popt, pcov = curve_fit(
#         exp_step_with_drift, tsec, y, p0=p0, bounds=bounds, maxfev=200000
#     )
#     T_inf, T0, t0_fit, tau_s, m = popt

#     y_fit = exp_step_with_drift(tsec, *popt)
#     ss_res = np.sum((y - y_fit) ** 2)
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

#     # Parameter uncertainties (1σ) from covariance
#     perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * len(popt)
#     keys = ["T_inf", "T0", "t0_s", "tau_s", "m_deg_per_s"]
#     vals = [T_inf, T0, t0_fit, tau_s, m]
#     errs = perr

#     # End-slope diagnostic (°C/min)
#     end_slope = end_slope_deg_per_min(dfw, tail_minutes=10)

#     if do_plot:
#         plt.figure(figsize=(6, 5))
#         plt.plot(dfw["Timestamp"], y, ".", label=f"{label} data")
#         plt.plot(
#             dfw["Timestamp"],
#             y_fit,
#             "-",
#             label=f"fit: τ={tau_s / 60:.1f} min, m={m * 60:.3f} °C/min, R²={r2:.3f}",
#         )
#         plt.title(f"{label} Thermal Step Fit (with drift)")
#         plt.xlabel("Time (local)")
#         plt.ylabel("Temperature [°C]")
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         # full datetime ticks, 45°
#         ax = plt.gca()
#         import matplotlib.dates as mdates

#         ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
#         ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
#         plt.tight_layout()
#         plt.show(block=True)

#     return {
#         "params": dict(zip(keys, vals)),
#         "perr": dict(zip(keys, errs)),
#         "tau_min": float(tau_s / 60.0),
#         "tau_hr": float(tau_s / 3600.0),
#         "r2": float(r2),
#         "end_slope_deg_per_min": float(end_slope),
#         "n_points": int(len(y)),
#     }


# def parse_local(ts_str: str) -> datetime.datetime:
#     """Parse 'YYYY-MM-DD-HH-MM-SS' as a naive local datetime."""
#     return datetime.datetime.strptime(ts_str, "%Y-%m-%d-%H-%M-%S")


# def get_channel_window_df(
#     label: str, t_start_local: str, t_end_local: str, t_low=temp_low, t_high=temp_high
# ) -> pd.DataFrame:
#     """
#     Load channel CSV (current+prev month), normalize, and slice to the given window.
#     t_start_local/t_end_local format: 'YYYY-MM-DD-HH-MM-SS' (local).
#     """
#     fname = channels.get(label)
#     if fname is None:
#         raise ValueError(
#             f"Unknown channel '{label}'. Available: {list(channels.keys())}"
#         )
#     df = load_channel_df(fname)
#     if df.empty:
#         return df
#     df = normalize_df(df).sort_values("Timestamp").copy()
#     # Sanity filter
#     df = df[(df["Temperature"] > t_low) & (df["Temperature"] < t_high)]
#     if df.empty:
#         return df

#     t0 = parse_local(t_start_local)
#     t1 = parse_local(t_end_local)
#     win = df[(df["Timestamp"] >= t0) & (df["Timestamp"] <= t1)].copy()
#     return win


# if __name__ == "__main__":
#     t_start = "2025-09-16-21-50-00"
#     t_end = "2025-09-16-23-18-00"
#     df_4d = get_channel_window_df("4D", t_start, t_end)
#     if df_4d.empty:
#         print("No data in window.")
#     else:
#         res = fit_time_constant_with_drift(df_4d, label="4D", do_plot=True)
#         print("\n--- Drift-Aware Time Constant Fit (4D) ---")
#         print(f"N points         : {res['n_points']}")
#         print(f"tau              : {res['tau_min']:.2f} min ({res['tau_hr']:.3f} h)")
#         print(
#             f"T0 / T_inf       : {res['params']['T0']:.3f} °C → {res['params']['T_inf']:.3f} °C"
#         )
#         print(
#             f"t0               : {res['params']['t0_s']:.2f} s (rel. to window start)"
#         )
#         print(f"m (drift)        : {res['params']['m_deg_per_s'] * 60:.4f} °C/min")
#         print(f"R^2              : {res['r2']:.4f}")
#         print(f"End-slope (tail) : {res['end_slope_deg_per_min']:.4f} °C/min")
#         print("1σ uncertainties :")
#         for k, v in res["perr"].items():
#             if np.isfinite(v):
#                 print(f"  {k}: ±{v:.3g}")
