import matplotlib.pyplot as plt
import numpy as np

from analysis import sc_rabi_analysis as ra
from utils import data_manager as dm
from utils import widefield
from utils import kplotlib as kpl
kpl.init_kplotlib()
file_stem = "2026_10_15-16_03_55-johnson-nv0_2026_10_02"
# file_stem = "2026_10_15-21_23_07-johnson-nv0_2026_10_02"

o2_inds = [
    1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15,
    20, 21, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    37, 38, 41, 44, 45, 46, 47, 50, 51, 53, 54,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
    68, 69, 71, 73, 74, 77, 81, 84, 85, 86,
    90, 92, 95, 96, 98, 101, 102, 105, 106, 107,
    108, 111, 112, 113, 114, 115, 117
]


data = dm.get_raw_data(
    file_stem=file_stem,
    load_npz=True,
)

nv_list = data["nv_list"]
taus = np.asarray(data["taus"])
counts = np.asarray(data["counts"])

avg_counts, avg_counts_ste = widefield.process_counts(
    nv_list,
    counts[0],
    counts[1],
    threshold=True,
)

avg_counts = np.asarray(avg_counts)
avg_counts_ste = np.asarray(avg_counts_ste)

# --------------------------------------------------
# Select only O2
# --------------------------------------------------

o2_counts = avg_counts[o2_inds]
o2_ste = avg_counts_ste[o2_inds]
o2_nv_list = [nv_list[ind] for ind in o2_inds]


# ==================================================
# MEAN trace
# ==================================================

mean_trace = np.nanmean(
    o2_counts,
    axis=0,
)

mean_sem = (
    np.nanstd(
        o2_counts,
        axis=0,
        ddof=1,
    )
    / np.sqrt(len(o2_inds))
)


# ==================================================
# MEDIAN trace
# ==================================================

median_trace = np.nanmedian(
    o2_counts,
    axis=0,
)

q25, q75 = np.nanpercentile(
    o2_counts,
    [25, 75],
    axis=0,
)


# ==================================================
# Fit the median
#
# fit_rabi_data already returns a median fit
# when passed the O2 subset.
# ==================================================

(
    fit_fns,
    popts,
    median_fit_fn,
    median_popt,
) = ra.fit_rabi_data(
    o2_nv_list,
    taus,
    o2_counts,
    o2_ste,
)

median_period = 1 / abs(median_popt[1])
median_pi = median_period / 2


# ==================================================
# Fit mean trace
# ==================================================

(
    mean_fit_fns,
    mean_popts,
    _,
    _,
) = ra.fit_rabi_data(
    [o2_nv_list[0]],
    taus,
    mean_trace[None, :],
    mean_sem[None, :],
)

mean_fit_fn = mean_fit_fns[0]
mean_popt = mean_popts[0]

mean_period = 1 / abs(mean_popt[1])
mean_pi = mean_period / 2


# ==================================================
# Plot
# ==================================================

dense_taus = np.linspace(
    taus.min(),
    taus.max(),
    1000,
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

# Mean
ax.errorbar(
    taus,
    mean_trace,
    yerr=mean_sem,
    fmt="o",
    capsize=2,
    label=(
        f"O2 mean, N={len(o2_inds)} "
        f"(TR={mean_period:.1f} ns, "
        f"pi={mean_pi:.1f} ns)"
    ),
)

ax.plot(
    dense_taus,
    mean_fit_fn(dense_taus),
    linewidth=2,
)

# Median
ax.plot(
    taus,
    median_trace,
    "s",
    label=(
        f"O2 median "
        f"(TR={median_period:.1f} ns, "
        f"pi={median_pi:.1f} ns)"
    ),
)

ax.fill_between(
    taus,
    q25,
    q75,
    alpha=0.15,
    label="O2 IQR",
)

ax.plot(
    dense_taus,
    median_fit_fn(dense_taus),
    "--",
    linewidth=2,
)

ax.set_xlabel(
    "Microwave pulse duration (ns)"
)

ax.set_ylabel(
    "Normalized signal"
)

ax.set_title(
    "O2 [11-1] Rabi — 2.7924 GHz"
)

ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()

print(
    f"Mean:   TR={mean_period:.2f} ns, "
    f"pi={mean_pi:.2f} ns"
)

print(
    f"Median: TR={median_period:.2f} ns, "
    f"pi={median_pi:.2f} ns"
)

plt.show(block=True)