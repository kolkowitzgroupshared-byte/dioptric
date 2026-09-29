# -*- coding: utf-8 -*-
"""
Widefield DEER-Hahn P1/RF pulse-position sweep.

Fixed signal conditions:
    P1/RF frequency = rf_freq_ghz (default example: 0.198 GHz)
    P1/RF duration  = rf_len_ns (default example: 400 ns)

Swept variable:
    RF pulse-center offset relative to the center of the NV Hahn pi pulse.

For each physical offset dT, acquire an interleaved pair:

    (dT, f_ON), (dT, f_OFF)

where
    f_ON  = fixed P1/JT resonance
    f_OFF = f_ON + ref_detuning_ghz

Thus the P1 timing is swept while the resonant frequency and pulse duration
remain fixed for the signal measurement.

Channel convention:
    uwave_ind_list[0] = selected NV source
    uwave_ind_list[1] = P1 / RF source

@author: schand
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils import widefield
from utils.constants import NVSig


def _quantize_4ns(values_ns):
    return np.asarray(
        [
            int(
                4
                * round(
                    float(val) / 4
                )
            )
            for val in values_ns
        ],
        dtype=int,
    )


def create_position_figure(
    rf_center_offset_ns,
    avg_contrast,
):
    """
    Quick raw ON-OFF DEER contrast plot.

    No per-NV median subtraction is applied.
    """
    avg_contrast = np.asarray(
        avg_contrast,
        dtype=float,
    )

    p25 = np.nanpercentile(
        avg_contrast,
        25,
        axis=0,
    )
    p50 = np.nanpercentile(
        avg_contrast,
        50,
        axis=0,
    )
    p75 = np.nanpercentile(
        avg_contrast,
        75,
        axis=0,
    )

    x_us = (
        np.asarray(
            rf_center_offset_ns,
            dtype=float,
        )
        / 1000.0
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for row in avg_contrast:
        ax.plot(
            x_us,
            row,
            linewidth=0.65,
            alpha=0.12,
        )

    ax.fill_between(
        x_us,
        p25,
        p75,
        alpha=0.22,
        label="IQR",
    )

    ax.plot(
        x_us,
        p50,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Median",
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.65,
        label="RF center = NV pi center",
    )

    ax.set_xlabel(
        "P1/RF pulse-center offset from NV pi (us)"
    )
    ax.set_ylabel(
        "Raw DEER contrast (ON - OFF)"
    )
    ax.set_title(
        "DEER-Hahn RF pulse-position sweep"
    )
    ax.legend()
    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    return (
        fig,
        p25,
        p50,
        p75,
    )


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    min_rf_center_offset_ns,
    max_rf_center_offset_ns,
    rf_freq_ghz=0.198,
    rf_len_ns=400,
    uwave_ind_list=[0, 1],
    tau_ns=18_000,
    nv_pi_ns=256,
    ref_detuning_ghz=0.6,
    dynamic_thresh=True,
):
    """
    Run fixed-frequency, fixed-duration DEER-Hahn RF-position sweep.

    Parameters
    ----------
    num_steps
        Number of PHYSICAL RF timing offsets.
        Acquisition uses twice this number because each timing point
        has ON and OFF frequency measurements.

    min_rf_center_offset_ns, max_rf_center_offset_ns
        RF pulse-center position relative to the center of the NV pi pulse.

        0:
            centers aligned
        negative:
            RF before NV pi
        positive:
            RF after NV pi

    rf_freq_ghz
        Fixed resonant P1 frequency for signal measurement.
        Example: 0.198 GHz.

    rf_len_ns
        Fixed P1 pulse duration.
        Example: 400 ns.
    """
    if len(uwave_ind_list) != 2:
        raise ValueError(
            "DEER-Hahn RF-position sweep requires "
            "exactly [NV_ind, RF_ind]."
        )

    nv_ind = uwave_ind_list[0]
    rf_ind = uwave_ind_list[1]

    pulse_gen = (
        tb.get_server_pulse_gen()
    )
    seq_file = (
        "deer_hahn_rf_position.py"
    )

    # ---------------------------------------------------------
    # Physical timing-offset axis
    # ---------------------------------------------------------
    rf_center_offset_ns = np.linspace(
        min_rf_center_offset_ns,
        max_rf_center_offset_ns,
        num_steps,
    )

    rf_center_offset_ns = (
        _quantize_4ns(
            rf_center_offset_ns
        )
    )

    # Remove accidental duplicates after 4 ns quantization.
    rf_center_offset_ns = np.unique(
        rf_center_offset_ns
    )

    if len(rf_center_offset_ns) < 2:
        raise ValueError(
            "Need at least two distinct 4-ns-quantized timing offsets."
        )

    # ---------------------------------------------------------
    # Validate timing window.
    # ---------------------------------------------------------
    rf_len_ns = int(
        4
        * round(
            float(rf_len_ns) / 4
        )
    )
    tau_ns = int(
        4
        * round(
            float(tau_ns) / 4
        )
    )
    nv_pi_ns = int(
        4
        * round(
            float(nv_pi_ns) / 4
        )
    )

    nominal_rf_start_ns = (
        tau_ns
        + (nv_pi_ns - rf_len_ns) / 2
    )

    sequence_end_ns = (
        2 * tau_ns
        + nv_pi_ns
    )

    rf_start_ns = (
        nominal_rf_start_ns
        + rf_center_offset_ns
    )
    rf_end_ns = (
        rf_start_ns
        + rf_len_ns
    )

    if np.min(rf_start_ns) < 0:
        raise ValueError(
            "Requested negative timing offset places RF pulse "
            "before the Hahn evolution window. "
            f"Minimum RF start = {np.min(rf_start_ns)} ns."
        )

    if np.max(rf_end_ns) > sequence_end_ns:
        raise ValueError(
            "Requested positive timing offset places RF pulse "
            "after the Hahn evolution window. "
            f"Maximum RF end = {np.max(rf_end_ns)} ns; "
            f"window end = {sequence_end_ns} ns."
        )

    num_position_points = len(
        rf_center_offset_ns
    )

    # ---------------------------------------------------------
    # Interleave ON/OFF at the SAME timing offset.
    #
    # offsets:
    #   [dT0, dT0, dT1, dT1, ...]
    #
    # RF frequency:
    #   [f_on, f_off, f_on, f_off, ...]
    # ---------------------------------------------------------
    rf_offset_interleaved = np.empty(
        2 * num_position_points,
        dtype=int,
    )

    rf_offset_interleaved[
        0::2
    ] = rf_center_offset_ns

    rf_offset_interleaved[
        1::2
    ] = rf_center_offset_ns

    rf_freq_off_ghz = (
        float(rf_freq_ghz)
        + float(ref_detuning_ghz)
    )

    rf_freq_interleaved = np.empty(
        2 * num_position_points,
        dtype=float,
    )

    rf_freq_interleaved[
        0::2
    ] = float(
        rf_freq_ghz
    )

    rf_freq_interleaved[
        1::2
    ] = rf_freq_off_ghz

    num_steps_actual = len(
        rf_offset_interleaved
    )

    print(
        "\nDEER-Hahn RF-position sweep"
        f"\n  NV source         : {nv_ind}"
        f"\n  P1/RF source      : {rf_ind}"
        f"\n  P1 ON frequency   : {1000*float(rf_freq_ghz):.3f} MHz"
        f"\n  P1 OFF frequency  : {1000*rf_freq_off_ghz:.3f} MHz"
        f"\n  P1 pulse length   : {rf_len_ns} ns"
        f"\n  tau               : {tau_ns/1000:.3f} us"
        f"\n  total free evo    : {2*tau_ns/1000:.3f} us"
        f"\n  NV pi             : {nv_pi_ns} ns"
        f"\n  position points   : {num_position_points}"
        f"\n  acquisition steps : {num_steps_actual}"
        f"\n  offset range      : "
        f"{rf_center_offset_ns.min()/1000:.3f} to "
        f"{rf_center_offset_ns.max()/1000:.3f} us"
        "\n  offset = 0        : RF center aligned with NV-pi center"
    )

    # ---------------------------------------------------------
    # Load QUA sequence for each run.
    # ---------------------------------------------------------
    def run_fn(step_inds):
        base_scc_args = (
            widefield.get_base_scc_seq_args(
                nv_list,
                uwave_ind_list,
            )
        )

        shuffled_offsets_ns = [
            int(
                rf_offset_interleaved[
                    ind
                ]
            )
            for ind in step_inds
        ]

        seq_args = [
            base_scc_args,
            shuffled_offsets_ns,
            int(rf_len_ns),
            int(tau_ns),
            int(nv_pi_ns),
        ]

        seq_args_string = (
            tb.encode_seq_args(
                seq_args
            )
        )

        pulse_gen.stream_load(
            seq_file,
            seq_args_string,
            num_reps,
        )

    # ---------------------------------------------------------
    # Configure MW sources for every acquisition step.
    # ---------------------------------------------------------
    def step_fn(step_ind):
        # NV source.
        nv_dict = (
            tb.get_virtual_sig_gen_dict(
                nv_ind
            )
        )
        nv_mw = (
            tb.get_server_sig_gen(
                nv_ind
            )
        )

        nv_mw.set_amp(
            nv_dict[
                "uwave_power"
            ]
        )
        nv_mw.set_freq(
            nv_dict[
                "frequency"
            ]
        )
        nv_mw.uwave_on()

        # P1/RF source.
        rf_dict = (
            tb.get_virtual_sig_gen_dict(
                rf_ind
            )
        )
        rf = (
            tb.get_server_sig_gen(
                rf_ind
            )
        )

        rf.set_amp(
            rf_dict[
                "uwave_power"
            ]
        )
        rf.set_freq(
            float(
                rf_freq_interleaved[
                    step_ind
                ]
            )
        )
        rf.uwave_on()

    # ---------------------------------------------------------
    # Acquire.
    # ---------------------------------------------------------
    raw_data = base_routine.main(
        nv_list,
        num_steps_actual,
        num_reps,
        num_runs,
        run_fn,
        step_fn,
        uwave_ind_list=uwave_ind_list,
        save_images=False,
        num_exps=1,
        ref_by_rep_parity=False,
    )

    raw_fig = None

    # ---------------------------------------------------------
    # Process interleaved ON/OFF pairs.
    # ---------------------------------------------------------
    try:
        counts = np.asarray(
            raw_data[
                "counts"
            ]
        )

        if counts.ndim == 5:
            counts_exp = counts[0]
        elif counts.ndim == 4:
            counts_exp = counts
        else:
            raise ValueError(
                "Unexpected counts shape: "
                f"{counts.shape}"
            )

        on_inds = np.arange(
            0,
            num_steps_actual,
            2,
        )
        off_inds = np.arange(
            1,
            num_steps_actual,
            2,
        )

        sig_counts = counts_exp[
            :,
            :,
            on_inds,
            :,
        ]

        ref_counts = counts_exp[
            :,
            :,
            off_inds,
            :,
        ]

        if dynamic_thresh:
            (
                sig_counts,
                ref_counts,
            ) = (
                widefield.threshold_counts(
                    nv_list,
                    sig_counts,
                    ref_counts,
                    dynamic_thresh=True,
                )
            )

        (
            avg_sig_counts,
            avg_sig_counts_ste,
            _,
        ) = widefield.average_counts(
            sig_counts
        )

        (
            avg_ref_counts,
            avg_ref_counts_ste,
            _,
        ) = widefield.average_counts(
            ref_counts
        )

        (
            avg_contrast,
            avg_contrast_ste,
        ) = widefield.calc_contrast(
            sig_counts,
            ref_counts,
        )

        (
            avg_snr,
            avg_snr_ste,
        ) = widefield.calc_snr(
            sig_counts,
            ref_counts,
        )

        (
            raw_fig,
            p25,
            p50,
            p75,
        ) = create_position_figure(
            rf_center_offset_ns,
            avg_contrast,
        )

        raw_data |= {
            "avg_sig_counts": np.ascontiguousarray(
                avg_sig_counts
            ),
            "avg_sig_counts_ste": np.ascontiguousarray(
                avg_sig_counts_ste
            ),
            "avg_ref_counts": np.ascontiguousarray(
                avg_ref_counts
            ),
            "avg_ref_counts_ste": np.ascontiguousarray(
                avg_ref_counts_ste
            ),
            "avg_contrast": np.ascontiguousarray(
                avg_contrast
            ),
            "avg_contrast_ste": np.ascontiguousarray(
                avg_contrast_ste
            ),
            "avg_snr": np.ascontiguousarray(
                avg_snr
            ),
            "avg_snr_ste": np.ascontiguousarray(
                avg_snr_ste
            ),
            "p25_raw_contrast": np.ascontiguousarray(
                p25
            ),
            "median_raw_contrast": np.ascontiguousarray(
                p50
            ),
            "p75_raw_contrast": np.ascontiguousarray(
                p75
            ),
        }

    except Exception:
        print(
            traceback.format_exc()
        )

    # ---------------------------------------------------------
    # Reset hardware.
    # ---------------------------------------------------------
    tb.reset_cfm()
    kpl.show()

    # ---------------------------------------------------------
    # Save with standard data manager.
    # ---------------------------------------------------------
    timestamp = (
        dm.get_time_stamp()
    )

    raw_data |= {
        "timestamp": timestamp,
        "experiment": "deer_hahn_rf_position",

        # Physical timing axis.
        "rf_center_offset_ns": np.ascontiguousarray(
            rf_center_offset_ns
        ),
        "rf_center_offset_units": "ns",

        # Actual acquisition-step arrays.
        "rf_center_offset_ns_interleaved": np.ascontiguousarray(
            rf_offset_interleaved
        ),
        "rf_freq_interleaved_ghz": np.ascontiguousarray(
            rf_freq_interleaved
        ),

        # Fixed RF settings.
        "rf_freq_ghz": float(
            rf_freq_ghz
        ),
        "rf_freq_off_ghz": float(
            rf_freq_off_ghz
        ),
        "ref_detuning_ghz": float(
            ref_detuning_ghz
        ),
        "rf_len_ns": int(
            rf_len_ns
        ),

        # Hahn settings.
        "tau_ns": int(
            tau_ns
        ),
        "total_free_evolution_ns": int(
            2 * tau_ns
        ),
        "nv_pi_ns": int(
            nv_pi_ns
        ),
        "dynamic_thresh": bool(
            dynamic_thresh
        ),

        # Timing convention.
        "rf_position_definition": (
            "RF pulse-center offset relative to NV-pi center; "
            "negative=before, zero=centered, positive=after"
        ),

        # Hardware roles.
        "nv_uwave_ind": int(
            nv_ind
        ),
        "rf_uwave_ind": int(
            rf_ind
        ),
    }

    repr_nv_sig = (
        widefield.get_repr_nv_sig(
            nv_list
        )
    )
    repr_nv_name = (
        repr_nv_sig.name
    )

    file_path = (
        dm.get_file_path(
            __file__,
            timestamp,
            repr_nv_name,
        )
    )

    keys_to_compress = [
        "rf_center_offset_ns",
        "rf_center_offset_ns_interleaved",
        "rf_freq_interleaved_ghz",
    ]

    for key in [
        "img_arrays",
        "avg_sig_counts",
        "avg_sig_counts_ste",
        "avg_ref_counts",
        "avg_ref_counts_ste",
        "avg_contrast",
        "avg_contrast_ste",
        "avg_snr",
        "avg_snr_ste",
        "p25_raw_contrast",
        "median_raw_contrast",
        "p75_raw_contrast",
    ]:
        if key in raw_data:
            keys_to_compress.append(
                key
            )

    dm.save_raw_data(
        raw_data,
        file_path,
        keys_to_compress,
    )

    if raw_fig is not None:
        dm.save_figure(
            raw_fig,
            file_path,
        )

    return raw_data


if __name__ == "__main__":
    kpl.init_kplotlib()

    # Suggested first run:
    #
    # main(
    #     nv_list=nv_list,
    #     num_steps=71,                 # ~500 ns spacing
    #     num_reps=2,
    #     num_runs=100,
    #     min_rf_center_offset_ns=-17_500,
    #     max_rf_center_offset_ns=17_500,
    #     rf_freq_ghz=0.198,            # 198 MHz
    #     rf_len_ns=400,                # fixed 400 ns pulse
    #     uwave_ind_list=[0, 1],
    #     tau_ns=18_000,
    #     nv_pi_ns=256,
    #     ref_detuning_ghz=0.6,
    #     dynamic_thresh=True,
    # )
    #
    plt.show(block=True)
