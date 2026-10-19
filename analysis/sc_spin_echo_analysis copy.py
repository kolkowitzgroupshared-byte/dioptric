# -*- coding: utf-8 -*-
"""
Combined spin-echo heatmap for all NVs.

Combines multiple spin-echo datasets along the run axis,
processes the counts using widefield.process_counts(),
and generates a single heatmap PDF/PNG.

No averaging trace.
No fitting.
No individual-NV pages.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# Add dioptric repository to Python path
# =============================================================================

# This works regardless of whether the Windows user is saroj, matth, etc.
# Script is assumed to live inside:
# ...\dioptric\analysis\your_script.py

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
from utils import widefield
from utils import kplotlib as kpl


# =============================================================================
# Kplotlib setup
# =============================================================================

# Uses the standard kplotlib configuration.
# In your kplotlib.py the default image colormap is "inferno".
kpl.init_kplotlib(force=True)

# Optional: we do not need interactive mode for this analysis script
plt.ioff()


# =============================================================================
# Data location
# =============================================================================

BASE = (
    r"\\192.168.0.197\G\nvdata"
    r"\pc_Purcell"
    r"\branch_master"
    r"\spin_echo"
    r"\2026_10"
)

# =============================================================================
# Spin-echo datasets to combine
# =============================================================================
STEMS = [
    "2026_10_16-07_30_05-johnson-nv0_2026_10_02",
    "2026_10_16-15_51_16-johnson-nv0_2026_10_02",
    "2026_10_17-01_32_33-johnson-nv0_2026_10_02",
    "2026_10_17-09_47_32-johnson-nv0_2026_10_02",
]



# =============================================================================
# Output filenames
# =============================================================================

OUT_PDF = os.path.join(
    BASE,
    "2026_10_16-johnson-spin-echo-combined-all-119NVs-heatmap.pdf",
)

OUT_PNG = os.path.join(
    BASE,
    "2026_10_16-johnson-spin-echo-combined-all-119NVs-heatmap.png",
)

OUT_TRACES_PDF = os.path.join(
    BASE,
    "2026_10_16-johnson-spin-echo-combined-all-119NVs-individual-traces.pdf",
)

# =============================================================================
# Load datasets
# =============================================================================

counts_list = []
metadata_list = []

for stem in STEMS:

    txt_path = os.path.join(
        BASE,
        stem + ".txt",
    )

    npz_path = os.path.join(
        BASE,
        stem + ".npz",
    )

    print()
    print("Loading:")
    print(stem)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    metadata_list.append(metadata)

    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    counts = np.load(
        npz_path,
        allow_pickle=False,
    )["counts"].astype(float)

    print(
        "Counts shape:",
        counts.shape,
    )

    counts_list.append(counts)


# =============================================================================
# Validate datasets before combining
# =============================================================================

meta0 = metadata_list[0]

num_nvs_ref = counts_list[0].shape[1]
num_steps_ref = counts_list[0].shape[3]
num_reps_ref = counts_list[0].shape[4]

taus_ref = np.asarray(
    meta0["taus"],
    dtype=float,
)


for ind, counts in enumerate(counts_list):

    if counts.shape[1] != num_nvs_ref:
        raise ValueError(
            f"Dataset {ind} has {counts.shape[1]} NVs, "
            f"expected {num_nvs_ref}."
        )

    if counts.shape[3] != num_steps_ref:
        raise ValueError(
            f"Dataset {ind} has {counts.shape[3]} steps, "
            f"expected {num_steps_ref}."
        )

    if counts.shape[4] != num_reps_ref:
        raise ValueError(
            f"Dataset {ind} has {counts.shape[4]} reps, "
            f"expected {num_reps_ref}."
        )

    taus_here = np.asarray(
        metadata_list[ind]["taus"],
        dtype=float,
    )

    if not np.allclose(
        taus_here,
        taus_ref,
    ):
        raise ValueError(
            f"Tau array in dataset {ind} does not match "
            "the first dataset."
        )

# =============================================================================
# Combine datasets
# =============================================================================

# Expected counts shape:
#
# (
#     experiment,   # signal / reference
#     NV,
#     run,
#     tau step,
#     repetition,
# )
#
# We concatenate only along the RUN dimension.

combined_counts = np.concatenate(
    counts_list,
    axis=2,
)


print()
print("======================================")
print("Combined data")
print("======================================")
print(
    "Combined counts shape:",
    combined_counts.shape,
)

print(
    "Number of NVs:",
    combined_counts.shape[1],
)

print(
    "Total runs:",
    combined_counts.shape[2],
)

print(
    "Number of tau points:",
    combined_counts.shape[3],
)

print(
    "Reps:",
    combined_counts.shape[4],
)


# =============================================================================
# Evolution-time axis
# =============================================================================

taus_ns = taus_ref

# Each stored tau is one arm of:
#
# pi/2 -- tau -- pi -- tau -- pi/2
#
# Therefore total free-evolution time is 2*tau.

t_us = (
    2.0
    * taus_ns
    / 1e3
)


# =============================================================================
# Process combined counts
# =============================================================================

nv_list = meta0["nv_list"]

norm_counts, norm_counts_ste = widefield.process_counts(
    nv_list,
    combined_counts[0],
    combined_counts[1],
    threshold=True,
)


num_nvs = norm_counts.shape[0]


print()
print(
    "Processed norm_counts shape:",
    norm_counts.shape,
)

# =============================================================================
# Quantify improvement with additional averaging
# =============================================================================

OUT_AVERAGING_PNG = os.path.join(
    BASE,
    "2026_10_17-johnson-spin-echo-averaging-convergence.png",
)

OUT_AVERAGING_PDF = os.path.join(
    BASE,
    "2026_10_17-johnson-spin-echo-averaging-convergence.pdf",
)

OUT_AVERAGING_CSV = os.path.join(
    BASE,
    "2026_10_17-johnson-spin-echo-averaging-convergence.csv",
)


def process_spin_echo_counts(counts):
    """Process one combined raw-count array."""
    return widefield.process_counts(
        nv_list,
        counts[0],
        counts[1],
        threshold=True,
    )


averaging_results = []


for num_datasets in range(1, len(counts_list) + 1):

    # -------------------------------------------------------------------------
    # Cumulative data:
    # dataset 1
    # datasets 1+2
    # datasets 1+2+3
    # datasets 1+2+3+4
    # -------------------------------------------------------------------------

    cumulative_counts = np.concatenate(
        counts_list[:num_datasets],
        axis=2,
    )

    num_runs_here = cumulative_counts.shape[2]


    # -------------------------------------------------------------------------
    # Process full cumulative dataset
    # -------------------------------------------------------------------------

    y, yerr = process_spin_echo_counts(
        cumulative_counts
    )


    # -------------------------------------------------------------------------
    # Metric 1:
    # Median pointwise SEM across all NVs and all tau values
    # -------------------------------------------------------------------------

    median_sem = np.nanmedian(
        yerr
    )


    # -------------------------------------------------------------------------
    # Metric 2:
    # SEM of the average-over-NVs trace
    #
    # This propagates each NV's uncertainty into the mean trace.
    # -------------------------------------------------------------------------

    mean_trace_sem = (
        np.sqrt(
            np.nansum(
                yerr ** 2,
                axis=0,
            )
        )
        / y.shape[0]
    )

    median_mean_trace_sem = np.nanmedian(
        mean_trace_sem
    )


    # -------------------------------------------------------------------------
    # Metric 3:
    # Even-run vs odd-run reproducibility
    #
    # This gives two interleaved, independent subsets.
    # It is relatively insensitive to slow chronological drift.
    # -------------------------------------------------------------------------

    counts_even = cumulative_counts[
        :,
        :,
        0::2,
        :,
        :
    ]

    counts_odd = cumulative_counts[
        :,
        :,
        1::2,
        :,
        :
    ]


    # Make them the same number of runs
    n_split = min(
        counts_even.shape[2],
        counts_odd.shape[2],
    )

    counts_even = counts_even[
        :,
        :,
        :n_split,
        :,
        :
    ]

    counts_odd = counts_odd[
        :,
        :,
        :n_split,
        :,
        :
    ]


    y_even, _ = process_spin_echo_counts(
        counts_even
    )

    y_odd, _ = process_spin_echo_counts(
        counts_odd
    )


    even_odd_rmse = np.sqrt(
        np.nanmean(
            (y_even - y_odd) ** 2
        )
    )


    # -------------------------------------------------------------------------
    # Metric 4:
    # First-half vs second-half reproducibility
    #
    # This is useful for detecting slow drift.
    # -------------------------------------------------------------------------

    half = (
        cumulative_counts.shape[2]
        // 2
    )


    first_half_counts = cumulative_counts[
        :,
        :,
        :half,
        :,
        :
    ]

    second_half_counts = cumulative_counts[
        :,
        :,
        -half:,
        :,
        :
    ]


    y_first, _ = process_spin_echo_counts(
        first_half_counts
    )

    y_second, _ = process_spin_echo_counts(
        second_half_counts
    )


    first_second_rmse = np.sqrt(
        np.nanmean(
            (y_first - y_second) ** 2
        )
    )


    averaging_results.append(
        {
            "num_datasets": num_datasets,
            "num_runs": num_runs_here,
            "median_sem": median_sem,
            "mean_trace_sem": median_mean_trace_sem,
            "even_odd_rmse": even_odd_rmse,
            "first_second_rmse": first_second_rmse,
        }
    )


# =============================================================================
# Compare with ideal 1/sqrt(N) averaging
# =============================================================================

sem0 = averaging_results[0]["median_sem"]
group_sem0 = averaging_results[0]["mean_trace_sem"]

runs0 = averaging_results[0]["num_runs"]


for result in averaging_results:

    runs = result["num_runs"]


    # Ideal SEM if everything is independent random noise
    ideal_scale = np.sqrt(
        runs0 / runs
    )


    result["ideal_sem"] = (
        sem0
        * ideal_scale
    )

    result["ideal_mean_trace_sem"] = (
        group_sem0
        * ideal_scale
    )


    # How much uncertainty has decreased
    result["sem_reduction_percent"] = (
        100.0
        * (
            1.0
            - result["median_sem"] / sem0
        )
    )


    # Equivalent SNR improvement
    result["snr_gain"] = (
        sem0
        / result["median_sem"]
    )


    # 1.0 means exactly ideal sqrt(N) behavior
    #
    # >1 means noisier than ideal
    # <1 means slightly better than the simple first-point prediction
    result["relative_to_ideal"] = (
        result["median_sem"]
        / result["ideal_sem"]
    )


# =============================================================================
# Print table
# =============================================================================

print()
print("=" * 95)
print("AVERAGING CONVERGENCE")
print("=" * 95)

header = (
    f"{'Sets':>4s} "
    f"{'Runs':>7s} "
    f"{'Median SEM':>12s} "
    f"{'Ideal SEM':>12s} "
    f"{'SEM red.':>10s} "
    f"{'SNR gain':>10s} "
    f"{'Actual/ideal':>13s} "
    f"{'EO RMSE':>11s} "
    f"{'Half RMSE':>11s}"
)

print(header)
print("-" * len(header))


for result in averaging_results:

    print(
        f"{result['num_datasets']:4d} "
        f"{result['num_runs']:7d} "
        f"{result['median_sem']:12.6f} "
        f"{result['ideal_sem']:12.6f} "
        f"{result['sem_reduction_percent']:9.1f}% "
        f"{result['snr_gain']:10.3f} "
        f"{result['relative_to_ideal']:13.3f} "
        f"{result['even_odd_rmse']:11.6f} "
        f"{result['first_second_rmse']:11.6f}"
    )


# =============================================================================
# Improvement added by each new dataset
# =============================================================================

print()
print("Incremental improvement:")

for ind in range(1, len(averaging_results)):

    previous = averaging_results[ind - 1]
    current = averaging_results[ind]

    improvement = (
        100.0
        * (
            previous["median_sem"]
            - current["median_sem"]
        )
        / previous["median_sem"]
    )

    print(
        f"{ind} -> {ind + 1} datasets: "
        f"median SEM decreased by {improvement:.1f}%"
    )


# =============================================================================
# Save CSV
# =============================================================================

import csv


with open(
    OUT_AVERAGING_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            averaging_results[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        averaging_results
    )


# =============================================================================
# Plot convergence
# =============================================================================

runs = np.array(
    [
        result["num_runs"]
        for result in averaging_results
    ]
)

measured_sem = np.array(
    [
        result["median_sem"]
        for result in averaging_results
    ]
)

ideal_sem = np.array(
    [
        result["ideal_sem"]
        for result in averaging_results
    ]
)

even_odd_rmse = np.array(
    [
        result["even_odd_rmse"]
        for result in averaging_results
    ]
)

first_second_rmse = np.array(
    [
        result["first_second_rmse"]
        for result in averaging_results
    ]
)


fig, axs = plt.subplots(
    1,
    2,
    figsize=(11, 4.8),
)


# -----------------------------------------------------------------------------
# SEM convergence
# -----------------------------------------------------------------------------

ax = axs[0]

ax.plot(
    runs,
    measured_sem,
    "o-",
    lw=1.2,
    ms=5,
    label="Measured median SEM",
)

ax.plot(
    runs,
    ideal_sem,
    "o--",
    lw=1.0,
    ms=4,
    label=r"Ideal $1/\sqrt{N}$",
)

ax.set_xlabel(
    "Cumulative runs"
)

ax.set_ylabel(
    "Median SEM"
)

ax.set_title(
    "Statistical precision"
)

ax.grid(
    alpha=0.15
)

ax.legend(
    fontsize=9
)


# -----------------------------------------------------------------------------
# Independent-subset reproducibility
# -----------------------------------------------------------------------------

ax = axs[1]

ax.plot(
    runs,
    even_odd_rmse,
    "o-",
    lw=1.2,
    ms=5,
    label="Even vs odd runs",
)

ax.plot(
    runs,
    first_second_rmse,
    "o-",
    lw=1.2,
    ms=5,
    label="First half vs second half",
)

ax.set_xlabel(
    "Cumulative runs"
)

ax.set_ylabel(
    "Trace RMSE"
)

ax.set_title(
    "Trace reproducibility"
)

ax.grid(
    alpha=0.15
)

ax.legend(
    fontsize=9
)


fig.tight_layout()


fig.savefig(
    OUT_AVERAGING_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_AVERAGING_PDF,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Averaging convergence PNG:")
print(OUT_AVERAGING_PNG)

print()
print("Averaging convergence PDF:")
print(OUT_AVERAGING_PDF)

print()
print("Averaging convergence CSV:")
print(OUT_AVERAGING_CSV)

# =============================================================================
# Generate heatmap
# =============================================================================

fig, ax = plt.subplots(
    figsize=(11, 8.5),
)


# IMPORTANT:
#
# No cmap is specified here.
#
# kpl.init_kplotlib() sets the default image colormap,
# which in your kplotlib.py is "inferno".

im = ax.imshow(
    norm_counts,
    origin="lower",
    aspect="auto",

    extent=[
        t_us.min(),
        t_us.max(),
        -0.5,
        num_nvs - 0.5,
    ],

    interpolation="nearest",

    # Fixed normalized-population scale
    vmin=0.0,
    vmax=1.0,
)


# =============================================================================
# Axes
# =============================================================================

ax.set_title(
    (
        "Johnson Spin Echo — Combined Data\n"
        f"{num_nvs} NVs, "
        f"{combined_counts.shape[2]} total runs"
    ),
    fontsize=16,
)


ax.set_xlabel(
    r"Total evolution time $2\tau$ ($\mu$s)",
)


ax.set_ylabel(
    "NV index",
)


# Nice x-axis ticks
ax.set_xticks(
    np.arange(
        0,
        t_us.max() + 1,
        20,
    )
)


# NV ticks every 10 NVs
ax.set_yticks(
    np.arange(
        0,
        num_nvs,
        10,
    )
)


ax.set_xlim(
    t_us.min(),
    t_us.max(),
)


ax.set_ylim(
    -0.5,
    num_nvs - 0.5,
)


# =============================================================================
# Color bar
# =============================================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
)


cbar.set_label(
    "Normalized population",
)


# Fixed ticks from 0 to 1
cbar.set_ticks(
    np.arange(
        0,
        1.01,
        0.2,
    )
)


# =============================================================================
# Save
# =============================================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)


fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# =============================================================================
# Optional Desktop copy
# =============================================================================

home = Path.home()

desktop_candidates = [
    home / "Desktop",
    home / "OneDrive" / "Desktop",
]

desktop = next(
    (
        path
        for path in desktop_candidates
        if path.exists()
    ),
    None,
)


if desktop is not None:

    desktop_pdf = (
        desktop
        / Path(OUT_PDF).name
    )

    shutil.copy2(
        OUT_PDF,
        desktop_pdf,
    )

    print()
    print(
        "Desktop PDF:",
        desktop_pdf,
    )

else:

    print()
    print(
        "Desktop folder not found; "
        "skipping Desktop copy."
    )


# =============================================================================
# Done
# =============================================================================

print()
print("======================================")
print("Finished")
print("======================================")

print()
print("PDF:")
print(OUT_PDF)

print()
print("PNG:")
print(OUT_PNG)

print()
print(
    f"Heatmap contains all {num_nvs} NVs."
)

print(
    "Color scale: 0 to 1"
)

print(
    "Colormap: kplotlib default"
)


# -*- coding: utf-8 -*-
"""
Combined spin-echo analysis.

Outputs:
1. Heatmap of all NVs using the two datasets combined.
2. Individual combined/averaged spin-echo traces for every NV.

The two datasets are concatenated along the run axis before
widefield.process_counts() is called.
"""

import os
import sys
import json
import math
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =============================================================================
# Repository
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

from utils import widefield
from utils import kplotlib as kpl


# =============================================================================
# Plot configuration
# =============================================================================

kpl.init_kplotlib(force=True)

# Not needed for a script that only saves figures
plt.ioff()


# =============================================================================
# Data
# =============================================================================

BASE = (
    r"\\192.168.0.197\G\nvdata"
    r"\pc_Purcell"
    r"\branch_master"
    r"\spin_echo"
    r"\2026_10"
)


STEMS = [
    "2026_10_16-07_30_05-johnson-nv0_2026_10_02",
    "2026_10_16-15_51_16-johnson-nv0_2026_10_02",
]


# =============================================================================
# Output files
# =============================================================================

OUT_HEATMAP_PDF = os.path.join(
    BASE,
    "2026_10_16-johnson-spin-echo-combined-all-119NVs-heatmap.pdf",
)

OUT_HEATMAP_PNG = os.path.join(
    BASE,
    "2026_10_16-johnson-spin-echo-combined-all-119NVs-heatmap.png",
)

OUT_TRACES_PDF = os.path.join(
    BASE,
    "2026_10_16-johnson-spin-echo-combined-all-119NVs-individual-traces.pdf",
)


# =============================================================================
# Load datasets
# =============================================================================

counts_list = []
metadata_list = []


for stem in STEMS:

    txt_path = os.path.join(
        BASE,
        stem + ".txt",
    )

    npz_path = os.path.join(
        BASE,
        stem + ".npz",
    )

    print()
    print("Loading:")
    print(stem)

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    counts = np.load(
        npz_path,
        allow_pickle=False,
    )["counts"].astype(float)

    print(
        "Counts shape:",
        counts.shape,
    )

    metadata_list.append(metadata)
    counts_list.append(counts)


# =============================================================================
# Validate datasets
# =============================================================================

meta0 = metadata_list[0]

reference_shape = counts_list[0].shape

num_nvs_ref = reference_shape[1]
num_steps_ref = reference_shape[3]
num_reps_ref = reference_shape[4]

taus_ref = np.asarray(
    meta0["taus"],
    dtype=float,
)


for dataset_ind, counts in enumerate(counts_list):

    if counts.shape[1] != num_nvs_ref:
        raise ValueError(
            f"Dataset {dataset_ind} has "
            f"{counts.shape[1]} NVs; "
            f"expected {num_nvs_ref}."
        )

    if counts.shape[3] != num_steps_ref:
        raise ValueError(
            f"Dataset {dataset_ind} has "
            f"{counts.shape[3]} tau points; "
            f"expected {num_steps_ref}."
        )

    if counts.shape[4] != num_reps_ref:
        raise ValueError(
            f"Dataset {dataset_ind} has "
            f"{counts.shape[4]} reps; "
            f"expected {num_reps_ref}."
        )

    taus_here = np.asarray(
        metadata_list[dataset_ind]["taus"],
        dtype=float,
    )

    if not np.allclose(
        taus_here,
        taus_ref,
    ):
        raise ValueError(
            f"Tau array for dataset {dataset_ind} "
            "does not match the first dataset."
        )


# =============================================================================
# Combine datasets
# =============================================================================

# Expected raw counts shape:
#
# (
#     signal/reference,
#     NV,
#     run,
#     tau step,
#     repetition,
# )
#
# Concatenate along the run axis.

combined_counts = np.concatenate(
    counts_list,
    axis=2,
)


print()
print("Combined counts shape:", combined_counts.shape)
print("Number of NVs:", combined_counts.shape[1])
print("Total runs:", combined_counts.shape[2])
print("Tau points:", combined_counts.shape[3])
print("Reps:", combined_counts.shape[4])


# =============================================================================
# Evolution-time axis
# =============================================================================

taus_ns = taus_ref

# Stored tau is one free-precession arm:
#
# pi/2 -- tau -- pi -- tau -- pi/2
#
# Total evolution time = 2*tau.

t_us = (
    2.0
    * taus_ns
    / 1e3
)


# =============================================================================
# Process combined counts
# =============================================================================

nv_list = meta0["nv_list"]


norm_counts, norm_counts_ste = widefield.process_counts(
    nv_list,
    combined_counts[0],
    combined_counts[1],
    threshold=True,
)


num_nvs = norm_counts.shape[0]


print()
print(
    "Processed norm_counts shape:",
    norm_counts.shape,
)


# =============================================================================
# Heatmap
# =============================================================================

fig, ax = plt.subplots(
    figsize=(11, 8.5),
)


# No cmap argument is supplied.
# kpl.init_kplotlib() determines the default image colormap.

im = ax.imshow(
    norm_counts,
    origin="lower",
    aspect="auto",

    extent=[
        t_us.min(),
        t_us.max(),
        -0.5,
        num_nvs - 0.5,
    ],

    interpolation="nearest",

    # Normalized population color range
    vmin=0.0,
    vmax=1.0,
)


ax.set_title(
    (
        "Johnson Spin Echo — Combined Data\n"
        f"{num_nvs} NVs, "
        f"{combined_counts.shape[2]} total runs"
    ),
    fontsize=15,
)


ax.set_xlabel(
    r"Total evolution time $2\tau$ ($\mu$s)"
)


ax.set_ylabel(
    "NV index"
)


ax.set_xticks(
    np.arange(
        0,
        t_us.max() + 1,
        20,
    )
)


ax.set_yticks(
    np.arange(
        0,
        num_nvs,
        10,
    )
)


ax.set_xlim(
    t_us.min(),
    t_us.max(),
)


ax.set_ylim(
    -0.5,
    num_nvs - 0.5,
)


# =============================================================================
# Heatmap colorbar
# =============================================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
)


cbar.set_label(
    "Normalized population"
)


cbar.set_ticks(
    np.arange(
        0,
        1.01,
        0.2,
    )
)


fig.savefig(
    OUT_HEATMAP_PDF,
    bbox_inches="tight",
)


fig.savefig(
    OUT_HEATMAP_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.close(fig)


# =============================================================================
# Individual averaged traces
# =============================================================================

# 6 NVs per page:
#
# 3 rows x 2 columns
#
# This gives enough space to inspect the revival structure.

nrows = 3
ncols = 2

per_page = (
    nrows
    * ncols
)

num_pages = math.ceil(
    num_nvs
    / per_page
)


with PdfPages(
    OUT_TRACES_PDF
) as pdf:

    for page_ind in range(num_pages):

        start_nv = (
            page_ind
            * per_page
        )

        stop_nv = min(
            start_nv + per_page,
            num_nvs,
        )


        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(11, 8.5),

            # Important:
            # share x axis, but NOT y axis,
            # because each NV is auto-scaled.
            sharex=True,
            sharey=False,
        )


        axs = np.asarray(
            axs
        ).ravel()


        # =====================================================================
        # NV traces
        # =====================================================================

        for plot_ind, ax in enumerate(axs):

            nv_ind = (
                start_nv
                + plot_ind
            )


            if nv_ind >= stop_nv:

                ax.axis(
                    "off"
                )

                continue


            # -----------------------------------------------------------------
            # Trace and uncertainty
            # -----------------------------------------------------------------

            y = np.asarray(
                norm_counts[nv_ind],
                dtype=float,
            )

            yerr = np.asarray(
                norm_counts_ste[nv_ind],
                dtype=float,
            )


            ax.errorbar(
                t_us,
                y,

                yerr=yerr,

                fmt="o-",

                # thin line
                lw=0.65,

                # thin error bars
                elinewidth=0.4,

                # small points
                ms=2.0,

                capsize=0.8,
                capthick=0.4,

                alpha=0.9,
            )


            # -----------------------------------------------------------------
            # Automatic y-scale
            #
            # Includes error bars so nothing gets clipped.
            # -----------------------------------------------------------------

            y_low = (
                y
                - yerr
            )

            y_high = (
                y
                + yerr
            )


            ymin = np.nanmin(
                y_low
            )

            ymax = np.nanmax(
                y_high
            )


            yrange = (
                ymax
                - ymin
            )


            # Protect against an almost-flat trace
            if (
                not np.isfinite(yrange)
                or yrange < 0.05
            ):
                yrange = 0.05


            # 8% margin
            margin = (
                0.08
                * yrange
            )


            lower = (
                ymin
                - margin
            )

            upper = (
                ymax
                + margin
            )


            # Round limits to the nearest 0.05
            # so axes look cleaner from panel to panel.

            step = 0.05


            lower = (
                np.floor(
                    lower / step
                )
                * step
            )


            upper = (
                np.ceil(
                    upper / step
                )
                * step
            )


            # Safety in the unlikely event that
            # rounded limits become identical.
            if upper <= lower:

                lower -= 0.05
                upper += 0.05


            ax.set_ylim(
                lower,
                upper,
            )


            # -----------------------------------------------------------------
            # X scale
            # -----------------------------------------------------------------

            ax.set_xlim(
                t_us.min(),
                t_us.max(),
            )


            ax.set_xticks(
                np.arange(
                    0,
                    t_us.max() + 1,
                    20,
                )
            )


            # -----------------------------------------------------------------
            # Formatting
            # -----------------------------------------------------------------

            ax.set_title(
                f"NV {nv_ind}",
                fontsize=11,
            )


            ax.grid(
                alpha=0.12,
                linewidth=0.45,
            )


            ax.tick_params(
                axis="both",
                labelsize=8,
            )


        # =====================================================================
        # Axis labels
        # =====================================================================

        for ax in axs[-ncols:]:

            if ax.axison:

                ax.set_xlabel(
                    r"Total evolution time $2\tau$ ($\mu$s)",
                    fontsize=9,
                )


        for ax in axs[::ncols]:

            if ax.axison:

                ax.set_ylabel(
                    "Normalized population",
                    fontsize=9,
                )


        # =====================================================================
        # Page title
        # =====================================================================

        fig.suptitle(
            (
                "Johnson Spin Echo — Combined Individual-NV Traces\n"
                f"NV {start_nv}–{stop_nv - 1}"
            ),
            fontsize=14,
        )


        fig.tight_layout(
            rect=[
                0.03,
                0.03,
                0.99,
                0.94,
            ],

            h_pad=1.6,
            w_pad=1.4,
        )


        pdf.savefig(
            fig,
            bbox_inches="tight",
        )


        plt.close(
            fig
        )


# =============================================================================
# Desktop copies
# =============================================================================

home = Path.home()


desktop_candidates = [
    home / "Desktop",
    home / "OneDrive" / "Desktop",
]


desktop = next(
    (
        path
        for path
        in desktop_candidates
        if path.exists()
    ),
    None,
)


if desktop is not None:

    desktop_heatmap_pdf = (
        desktop
        / Path(
            OUT_HEATMAP_PDF
        ).name
    )

    desktop_traces_pdf = (
        desktop
        / Path(
            OUT_TRACES_PDF
        ).name
    )


    shutil.copy2(
        OUT_HEATMAP_PDF,
        desktop_heatmap_pdf,
    )


    shutil.copy2(
        OUT_TRACES_PDF,
        desktop_traces_pdf,
    )


    print()
    print(
        "Desktop heatmap PDF:"
    )

    print(
        desktop_heatmap_pdf
    )


    print()
    print(
        "Desktop traces PDF:"
    )

    print(
        desktop_traces_pdf
    )


else:

    print()
    print(
        "Desktop folder not found."
    )

    print(
        "Skipping Desktop copies."
    )


# =============================================================================
# Finished
# =============================================================================

print()
print("======================================")
print("Finished")
print("======================================")


print()
print("Heatmap PDF:")
print(
    OUT_HEATMAP_PDF
)


print()
print("Heatmap PNG:")
print(
    OUT_HEATMAP_PNG
)


print()
print("Individual traces PDF:")
print(
    OUT_TRACES_PDF
)


print()
print(
    f"Number of NVs: {num_nvs}"
)

print(
    f"Total combined runs: "
    f"{combined_counts.shape[2]}"
)