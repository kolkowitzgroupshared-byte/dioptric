# -*- coding: utf-8 -*-
"""
Lightweight check of the SCC SNR.

Acquires one-point SCC SNR data and immediately runs the same analysis used by
analysis/sc_scc_snr_check_analysis.py.

Important:
dm.save_raw_data() currently replaces data["counts"] in-place with an NPZ-link
string. Therefore analysis MUST be done before saving the raw data.

Created December 6th, 2023
Updated October 2026

@author: mccambria
@author: schand
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np

from analysis import sc_scc_snr_check_analysis as scc_analysis
from majorroutines.widefield import base_routine
from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield


SAVE_MAIN_FIGURE = True
SAVE_JOINT_FIGURE = True
SAVE_ANALYSIS_SUMMARY = True
SHOW_FIGURES = True


def process_and_plot(data):
    """
    Run the same SCC-SNR analysis/plots used by
    analysis/sc_scc_snr_check_analysis.py.

    Returns
    -------
    figures : dict
        {"snr": fig, "amp_duration_snr": joint_fig}
    df : pandas.DataFrame
        Per-NV numerical results.
    metadata : dict
        Summary metadata.
    """

    counts_obj = data.get("counts", None)

    if counts_obj is None:
        raise ValueError("data['counts'] is None.")

    if isinstance(counts_obj, (str, bytes)):
        raise ValueError(
            "data['counts'] is already an NPZ-link string. "
            "Run process_and_plot(data) BEFORE dm.save_raw_data(data, ...)."
        )

    counts = np.asarray(counts_obj)
    if counts.ndim < 2:
        raise ValueError(
            f"Expected counts array with signal/reference axis; got shape {counts.shape}. "
            "This usually means dm.save_raw_data() was called first."
        )

    # The offline analysis expects opx_config in the data dictionary.
    # During live acquisition it has not yet been inserted by save_raw_data(),
    # so provide the current config explicitly.
    if data.get("opx_config") is None:
        data["opx_config"] = common.get_opx_config_dict()

    # Keep live analysis matched to the standalone analysis file.
    scc_analysis.X_AXIS = "scc_duration"
    scc_analysis.STEP_IND = 0
    scc_analysis.APPLY_THRESHOLD = True

    df, metadata = scc_analysis.process_data(data)

    print()
    print(
        df[
            [
                "NV Index",
                "SCC AOD Multiplier",
                "SCC Duration (ns)",
                "SNR",
                "SNR STE",
            ]
        ]
    )

    fig = scc_analysis.plot_snr(df, metadata)
    joint_fig = scc_analysis.plot_amp_duration_snr(df, metadata)

    figures = {
        "snr": fig,
        "amp_duration_snr": joint_fig,
    }

    return figures, df, metadata


def _save_analysis_summary(
    df,
    metadata,
    timestamp,
    repr_nv_name,
    raw_file_path,
    num_reps,
    num_runs,
    uwave_ind_list,
):
    results = {
        "timestamp": timestamp,
        "source_file_stem": raw_file_path.stem,
        "num_nvs": int(metadata["num_nvs"]),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind_list": list(uwave_ind_list),
        "nv_index": df["NV Index"].tolist(),
        "scc_aod_multiplier": df["SCC AOD Multiplier"].tolist(),
        "scc_duration_ns": df["SCC Duration (ns)"].tolist(),
        "snr": df["SNR"].tolist(),
        "snr_ste": df["SNR STE"].tolist(),
        "contrast": df["Contrast"].tolist(),
        "contrast_ste": df["Contrast STE"].tolist(),
        "signal_counts": df["Signal Counts"].tolist(),
        "reference_counts": df["Reference Counts"].tolist(),
        "readout_power_uw": float(metadata["readout_power_uw"]),
        "spin_pol_power_uw": float(metadata["spin_pol_power_uw"]),
        "median_snr": float(np.nanmedian(df["SNR"])),
        "mean_snr": float(np.nanmean(df["SNR"])),
    }

    analysis_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name + "-analysis",
    )

    dm.save_raw_data(results, analysis_path)
    print(f"Saved analysis summary: {analysis_path}")


def main(nv_list, num_reps, num_runs, uwave_ind_list):
    num_steps = 1

    seq_file = "scc_snr_check.py"
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(
                nv_list,
                uwave_ind_list,
            ),
        ]
        seq_args_string = tb.encode_seq_args(seq_args)
        pulse_gen.stream_load(
            seq_file,
            seq_args_string,
            num_reps,
        )

    # ------------------------------------------------------------------
    # Acquire
    # ------------------------------------------------------------------
    data = base_routine.main(
        nv_list,
        num_steps,
        num_reps,
        num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        charge_prep_fn=None,
        num_exps=2,
        load_iq=True,
    )

    timestamp = dm.get_time_stamp()
    repr_nv_name = widefield.get_repr_nv_sig(nv_list).name
    raw_file_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name,
    )

    # ------------------------------------------------------------------
    # Analyze FIRST.
    #
    # save_raw_data() mutates data["counts"] into an NPZ-link string,
    # so doing this after save_raw_data() causes the 0-D IndexError.
    # ------------------------------------------------------------------
    figures = {}
    df = None
    metadata = None

    try:
        figures, df, metadata = process_and_plot(data)
    except Exception:
        print()
        print("SCC SNR live analysis failed:")
        print(traceback.format_exc())

    # ------------------------------------------------------------------
    # Save raw data AFTER the analysis has consumed the real counts array.
    # ------------------------------------------------------------------
    dm.save_raw_data(data, raw_file_path)
    print(f"Saved raw data: {raw_file_path}")

    # ------------------------------------------------------------------
    # Save figures beside the raw dataset.
    # ------------------------------------------------------------------
    if figures:
        if SAVE_MAIN_FIGURE and "snr" in figures:
            fig_path = dm.get_file_path(
                __file__,
                timestamp,
                repr_nv_name + "-snr",
            )
            dm.save_figure(figures["snr"], fig_path)
            print(f"Saved SNR figure: {fig_path}")

        if SAVE_JOINT_FIGURE and "amp_duration_snr" in figures:
            joint_path = dm.get_file_path(
                __file__,
                timestamp,
                repr_nv_name + "-amp-duration-snr",
            )
            dm.save_figure(
                figures["amp_duration_snr"],
                joint_path,
            )
            print(f"Saved joint figure: {joint_path}")

    # ------------------------------------------------------------------
    # Save compact processed values too.
    # ------------------------------------------------------------------
    if (
        SAVE_ANALYSIS_SUMMARY
        and df is not None
        and metadata is not None
    ):
        _save_analysis_summary(
            df,
            metadata,
            timestamp,
            repr_nv_name,
            raw_file_path,
            num_reps,
            num_runs,
            uwave_ind_list,
        )

    tb.reset_cfm()

    if SHOW_FIGURES and figures:
        kpl.show(block=True)
    else:
        plt.close("all")

    return data


if __name__ == "__main__":
    kpl.init_kplotlib()
