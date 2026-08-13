from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import data_manager as dm
from utils import kplotlib as kpl


def export_49G_selected_traces(
    nv_inds=(137, 172, 196),
    output_dir="shared_49G_traces",
    use_tau=True,
):
    """
    Export selected 49 G spin-echo traces.

    Parameters
    ----------
    nv_inds:
        NV indices to export.

    output_dir:
        Folder for CSV, PNG, and PDF files.

    use_tau:
        True:
            x axis = tau = total evolution time / 2.

        False:
            x axis = total evolution time.
    """

    counts_file_stem = "2025_11_11-01_15_45-" "johnson_204nv_s6-6d8f5c"

    data = dm.get_raw_data(file_stem=counts_file_stem)

    norm_counts = np.asarray(
        data["norm_counts"],
        dtype=float,
    )

    norm_counts_ste = np.asarray(
        data["norm_counts_ste"],
        dtype=float,
    )

    total_time_us = np.asarray(
        data["total_evolution_times"],
        dtype=float,
    )

    if use_tau:
        x_us = total_time_us / 2.0
        x_name = "tau_us"
        x_label = r"$\tau$ ($\mu$s)"
    else:
        x_us = total_time_us
        x_name = "total_evolution_time_us"
        x_label = r"Total evolution time ($\mu$s)"

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nv_inds = [int(nv_ind) for nv_ind in nv_inds]

    # --------------------------------------------------------------
    # Validate indices
    # --------------------------------------------------------------

    num_nvs = norm_counts.shape[0]

    for nv_ind in nv_inds:
        if not 0 <= nv_ind < num_nvs:
            raise IndexError(
                f"NV {nv_ind} is outside the available " f"range 0–{num_nvs - 1}."
            )

    # --------------------------------------------------------------
    # Build simple long-format CSV
    # --------------------------------------------------------------

    table_rows = []

    for nv_ind in nv_inds:
        signal = norm_counts[nv_ind]
        signal_ste = norm_counts_ste[nv_ind]

        for x_value, y_value, y_error in zip(
            x_us,
            signal,
            signal_ste,
        ):
            table_rows.append(
                {
                    "field_G": 49,
                    "nv_index": nv_ind,
                    x_name: x_value,
                    "normalized_signal": y_value,
                    "signal_ste": y_error,
                }
            )

    table = pd.DataFrame(table_rows)

    csv_path = output_dir / ("49G_NV137_NV172_NV196_traces.csv")

    table.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Three-panel figure
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        len(nv_inds),
        1,
        figsize=(8, 2.8 * len(nv_inds)),
        sharex=True,
        constrained_layout=True,
    )

    if len(nv_inds) == 1:
        axes = [axes]

    for ax, nv_ind in zip(
        axes,
        nv_inds,
    ):
        signal = norm_counts[nv_ind]
        signal_ste = norm_counts_ste[nv_ind]

        good = np.isfinite(x_us) & np.isfinite(signal) & np.isfinite(signal_ste)

        ax.errorbar(
            x_us[good],
            signal[good],
            yerr=signal_ste[good],
            marker="o",
            markersize=3,
            linestyle="-",
            linewidth=1,
            capsize=2,
        )

        ax.set_ylabel("Normalized\nsignal")

        ax.set_title(f"NV {nv_ind}")

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[-1].set_xlabel(x_label)

    fig.suptitle(
        "Selected spin-echo traces at 49 G",
        fontsize=14,
    )

    png_path = output_dir / ("49G_NV137_NV172_NV196_traces.png")

    pdf_path = output_dir / ("49G_NV137_NV172_NV196_traces.pdf")

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("Saved CSV:", csv_path)
    print("Saved PNG:", png_path)
    print("Saved PDF:", pdf_path)

    return table, fig


if __name__ == "__main__":
    kpl.init_kplotlib()

    table, fig = export_49G_selected_traces(
        nv_inds=[137, 172, 196],
        output_dir="shared_49G_traces",
        use_tau=True,
    )

    plt.show()
