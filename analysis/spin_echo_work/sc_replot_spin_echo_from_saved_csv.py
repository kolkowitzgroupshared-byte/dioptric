# -*- coding: utf-8 -*-
"""
Recover all spin-echo plots from an already-saved fit_results.csv.

NO nonlinear fitting is performed.

This is intended for the run that completed all 212 fits but crashed during
saving because dm.save_raw_data() received a string instead of a Path.

It reconstructs the core/ESEEM curves from the fitted CSV parameters and the
saved combined normalized dataset, then writes:
    *_summary.png
    *_summary.pdf
    *_all_nv_full_fits.pdf
    *_all_nv_first_revival_fits.pdf
    *_reconstructed_fit_checkpoint.npz
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import kplotlib as kpl

# Import your fitting module only for its model/plot helper functions.
# Importing it does NOT execute main().
from analysis.spin_echo_work import (
    sc_c13_spin_echo_physics_fit_52G_nv_pillar_array as fitmod,
)


# =============================================================================
# USER SETTINGS
# =============================================================================

FIT_RESULTS_CSV = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo"
    r"\sc_c13_spin_echo_physics_fit_52G_nv_pillar_array"
    r"\2026_09"
    r"\2026_09_21-15_48_27-spin_echo_physics_fit_52G_fit_results.csv"
)

SHOW_SUMMARY = True


# =============================================================================
# HELPERS
# =============================================================================

def _as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    if pd.isna(value):
        return False

    return bool(value)


def _finite(row, key):
    try:
        return np.isfinite(
            float(row[key])
        )
    except Exception:
        return False


def reconstruct_result(
    row,
    tau_us,
):
    """
    Reconstruct fit_curve/core_curve from CSV parameters.
    """
    nv_index = int(
        row["nv_index"]
    )

    core_p = np.array(
        [
            float(row["baseline"]),
            float(row["contrast"]),
            float(row["revival_tau_us"]),
            float(row["width_us"]),
            float(row["T2_us"]),
            float(row["T2_exp"]),
            float(row["taper_alpha"]),
        ],
        dtype=float,
    )

    core_curve = fitmod.core_model(
        tau_us,
        *core_p,
    )

    fit_curve = core_curve.copy()

    eseem_used = _as_bool(
        row.get(
            "eseem_used",
            False,
        )
    )

    if (
        eseem_used
        and _finite(row, "f_minus_kHz")
        and _finite(row, "f_plus_kHz")
        and _finite(row, "amp_minus")
        and _finite(row, "phase_minus_rad")
        and _finite(row, "amp_plus")
        and _finite(row, "phase_plus_rad")
    ):
        amp_minus = float(
            row["amp_minus"]
        )
        phi_minus = float(
            row["phase_minus_rad"]
        )

        amp_plus = float(
            row["amp_plus"]
        )
        phi_plus = float(
            row["phase_plus_rad"]
        )

        # Original fitter uses:
        #   phase = atan2(-s, c)
        # therefore:
        #   c = A cos(phi)
        #   s = -A sin(phi)
        quad = np.array(
            [
                amp_minus
                * np.cos(phi_minus),
                -amp_minus
                * np.sin(phi_minus),
                amp_plus
                * np.cos(phi_plus),
                -amp_plus
                * np.sin(phi_plus),
            ],
            dtype=float,
        )

        carrier = fitmod.core_carrier(
            tau_us,
            core_p[2],
            core_p[3],
            core_p[4],
            core_p[5],
            core_p[6],
        )

        X = fitmod.design_eseem_matrix(
            tau_us,
            carrier,
            float(
                row["f_minus_kHz"]
            ),
            float(
                row["f_plus_kHz"]
            ),
        )

        fit_curve = (
            core_curve
            + X @ quad
        )

    # Only the fields used by the plotting helpers are required.
    return SimpleNamespace(
        nv_index=nv_index,
        status=str(
            row.get(
                "status",
                "reconstructed",
            )
        ),
        red_chi2=float(
            row.get(
                "red_chi2",
                np.nan,
            )
        ),
        eseem_used=eseem_used,
        delta_aicc=float(
            row.get(
                "delta_aicc",
                np.nan,
            )
        ),
        f_minus_kHz=float(
            row.get(
                "f_minus_kHz",
                np.nan,
            )
        ),
        f_plus_kHz=float(
            row.get(
                "f_plus_kHz",
                np.nan,
            )
        ),
        fit_curve=np.asarray(
            fit_curve,
            dtype=float,
        ),
        core_curve=np.asarray(
            core_curve,
            dtype=float,
        ),
    )


def main():
    kpl.init_kplotlib()

    if not FIT_RESULTS_CSV.exists():
        raise FileNotFoundError(
            FIT_RESULTS_CSV
        )

    print("=" * 78)
    print("REPLOT FROM SAVED FIT RESULTS — NO FITTING")
    print("=" * 78)
    print(f"CSV: {FIT_RESULTS_CSV}")
    print(
        f"Combined data stem from fitter: "
        f"{fitmod.FILE_STEM}"
    )

    # The fitter's FILE_STEM should point to the compact combined processed
    # dataset that was already saved before fitting.
    (
        _data,
        _nv_list,
        tau_us,
        _total_evolution_us,
        norm_counts,
        norm_counts_ste,
        _orientations,
    ) = fitmod.load_single_file(
        fitmod.FILE_STEM
    )

    df = pd.read_csv(
        FIT_RESULTS_CSV
    )

    results = [
        reconstruct_result(
            row,
            tau_us,
        )
        for _, row in df.iterrows()
    ]

    # Ensure plot order follows NV index.
    results.sort(
        key=lambda r: r.nv_index
    )

    output_base = Path(
        str(FIT_RESULTS_CSV)
        .replace(
            "_fit_results.csv",
            "",
        )
    )

    # Save reconstructed checkpoint so subsequent replotting does not even
    # need to reconstruct curves from the CSV.
    fit_curves = np.vstack(
        [
            r.fit_curve
            for r in results
        ]
    )

    core_curves = np.vstack(
        [
            r.core_curve
            for r in results
        ]
    )

    checkpoint_path = Path(
        str(output_base)
        + "_reconstructed_fit_checkpoint.npz"
    )

    np.savez_compressed(
        checkpoint_path,
        source_file_stem=np.asarray(
            [str(fitmod.FILE_STEM)]
        ),
        tau_us=np.asarray(
            tau_us,
            dtype=float,
        ),
        total_evolution_us=(
            2.0
            * np.asarray(
                tau_us,
                dtype=float,
            )
        ),
        norm_counts=np.asarray(
            norm_counts,
            dtype=float,
        ),
        norm_counts_ste=np.asarray(
            norm_counts_ste,
            dtype=float,
        ),
        nv_indices=np.asarray(
            [
                r.nv_index
                for r in results
            ],
            dtype=int,
        ),
        fit_curves=fit_curves,
        core_curves=core_curves,
    )

    print(
        f"Saved: {checkpoint_path}"
    )

    summary_fig = (
        fitmod.make_summary_figure(
            tau_us,
            norm_counts,
            norm_counts_ste,
            results,
        )
    )

    summary_png = Path(
        str(output_base)
        + "_summary.png"
    )

    summary_pdf = Path(
        str(output_base)
        + "_summary.pdf"
    )

    summary_fig.savefig(
        summary_png,
        dpi=300,
        bbox_inches="tight",
    )

    summary_fig.savefig(
        summary_pdf,
        bbox_inches="tight",
    )

    print(
        f"Saved: {summary_png}"
    )
    print(
        f"Saved: {summary_pdf}"
    )

    full_pdf = Path(
        str(output_base)
        + "_all_nv_full_fits.pdf"
    )

    revival_pdf = Path(
        str(output_base)
        + "_all_nv_first_revival_fits.pdf"
    )

    fitmod.save_fit_pdf(
        full_pdf,
        tau_us,
        norm_counts,
        norm_counts_ste,
        results,
        zoom_first_revival=False,
    )

    fitmod.save_fit_pdf(
        revival_pdf,
        tau_us,
        norm_counts,
        norm_counts_ste,
        results,
        zoom_first_revival=True,
    )

    print()
    print("=" * 78)
    print("REPLOT COMPLETE — NO FITTING WAS RUN")
    print("=" * 78)

    if SHOW_SUMMARY:
        plt.show(
            block=True
        )
    else:
        plt.close(
            summary_fig
        )


if __name__ == "__main__":
    main()
