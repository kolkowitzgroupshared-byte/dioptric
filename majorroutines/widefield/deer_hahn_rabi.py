# -*- coding: utf-8 -*-
"""
Widefield DEER-Hahn Rabi acquisition.

This is the Rabi counterpart of the working deer_hahn.py frequency scan.

For each physical P1 pulse duration L, acquire an interleaved pair:

    (L, f_ON), (L, f_OFF)

where
    f_ON  = fixed P1/JT resonance,
    f_OFF = f_ON + ref_detuning_ghz.

Thus the P1 frequency is fixed at one selected resonance for the signal
measurement, while a fixed detuned frequency provides the reference.
Only the P1 pulse duration is swept.

Channel convention:
    uwave_ind_list[0] = selected NV control source
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
            int(4 * round(float(val) / 4))
            for val in values_ns
        ],
        dtype=int,
    )


def create_deer_rabi_figure(
    rf_len_ns,
    avg_contrast,
):
    """Quick median/IQR DEER-Rabi plot."""
    avg_contrast = np.asarray(
        avg_contrast,
        dtype=float,
    )

    baseline = np.nanmedian(
        avg_contrast,
        axis=1,
        keepdims=True,
    )
    centered = avg_contrast - baseline

    p25 = np.nanpercentile(
        centered,
        25,
        axis=0,
    )
    p50 = np.nanpercentile(
        centered,
        50,
        axis=0,
    )
    p75 = np.nanpercentile(
        centered,
        75,
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    for row in centered:
        ax.plot(
            rf_len_ns,
            row,
            linewidth=0.7,
            alpha=0.15,
        )

    ax.fill_between(
        rf_len_ns,
        p25,
        p75,
        alpha=0.22,
        label="IQR",
    )

    ax.plot(
        rf_len_ns,
        p50,
        marker="o",
        markersize=3,
        linewidth=1.4,
        label="Median",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    ax.set_xlabel(
        "P1 / RF pulse duration (ns)"
    )
    ax.set_ylabel(
        "Baseline-subtracted DEER contrast"
    )
    ax.set_title(
        "DEER-Hahn Rabi"
    )
    ax.legend()
    ax.grid(alpha=0.15)

    fig.tight_layout()

    return fig, centered, p25, p50, p75


def main(
    nv_list: list[NVSig],
    num_steps,
    num_reps,
    num_runs,
    min_rf_len_ns,
    max_rf_len_ns,
    rf_freq_ghz,
    uwave_ind_list=[0, 1],
    tau_ns=18_000,
    nv_pi_ns=256,
    ref_detuning_ghz=0.6,
    center_rf_on_nv_pi=True,
    dynamic_thresh=True,
):
    """
    Run fixed-frequency P1 DEER Rabi.

    Parameters
    ----------
    num_steps
        Number of PHYSICAL Rabi pulse durations. Acquisition uses twice
        this number because every duration has ON and OFF frequency steps.
    rf_freq_ghz
        Resonant P1/JT frequency for the ON measurement, in GHz.
    ref_detuning_ghz
        OFF reference frequency shift, matching deer_hahn.py.
    """
    if len(uwave_ind_list) != 2:
        raise ValueError(
            "DEER-Hahn Rabi requires exactly "
            "[NV_ind, RF_ind]."
        )

    nv_ind = uwave_ind_list[0]
    rf_ind = uwave_ind_list[1]

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "deer_hahn_rabi.py"

    # ---------------------------------------------------------
    # Physical Rabi axis
    # ---------------------------------------------------------
    rf_len_ns = np.linspace(
        min_rf_len_ns,
        max_rf_len_ns,
        num_steps,
    )
    rf_len_ns = _quantize_4ns(
        rf_len_ns
    )

    # Remove accidental duplicates after 4 ns quantization.
    rf_len_ns = np.unique(rf_len_ns)

    if len(rf_len_ns) < 2:
        raise ValueError(
            "Need at least two distinct 4-ns-quantized RF durations."
        )

    if np.any(rf_len_ns <= 0):
        raise ValueError(
            "All RF pulse durations must be > 0 ns."
        )

    num_rabi_points = len(rf_len_ns)

    # ---------------------------------------------------------
    # Interleave ON/OFF at the SAME pulse duration.
    #
    # pulse length:
    #   [L0, L0, L1, L1, ...]
    #
    # RF frequency:
    #   [f_on, f_off, f_on, f_off, ...]
    # ---------------------------------------------------------
    rf_len_interleaved = np.empty(
        2 * num_rabi_points,
        dtype=int,
    )
    rf_len_interleaved[0::2] = rf_len_ns
    rf_len_interleaved[1::2] = rf_len_ns

    rf_freq_off_ghz = (
        float(rf_freq_ghz)
        + float(ref_detuning_ghz)
    )

    rf_freq_interleaved = np.empty(
        2 * num_rabi_points,
        dtype=float,
    )
    rf_freq_interleaved[0::2] = float(
        rf_freq_ghz
    )
    rf_freq_interleaved[1::2] = (
        rf_freq_off_ghz
    )

    num_steps_actual = len(
        rf_len_interleaved
    )

    print(
        "\nDEER-Hahn Rabi acquisition"
        f"\n  NV source       : {nv_ind}"
        f"\n  P1/RF source    : {rf_ind}"
        f"\n  P1 ON frequency : {1000*float(rf_freq_ghz):.3f} MHz"
        f"\n  P1 OFF frequency: {1000*rf_freq_off_ghz:.3f} MHz"
        f"\n  tau             : {tau_ns/1000:.3f} us"
        f"\n  total free evo  : {2*tau_ns/1000:.3f} us"
        f"\n  NV pi           : {nv_pi_ns} ns"
        f"\n  Rabi points     : {num_rabi_points}"
        f"\n  total steps     : {num_steps_actual}"
        f"\n  RF length range : "
        f"{rf_len_ns.min()}-{rf_len_ns.max()} ns"
    )

    # ---------------------------------------------------------
    # Load QUA sequence for each run
    # ---------------------------------------------------------
    def run_fn(step_inds):
        base_scc_args = (
            widefield.get_base_scc_seq_args(
                nv_list,
                uwave_ind_list,
            )
        )

        shuffled_rf_len_ns = [
            int(rf_len_interleaved[ind])
            for ind in step_inds
        ]

        seq_args = [
            base_scc_args,
            shuffled_rf_len_ns,
            int(tau_ns),
            int(nv_pi_ns),
            bool(center_rf_on_nv_pi),
        ]

        seq_args_string = tb.encode_seq_args(
            seq_args
        )

        pulse_gen.stream_load(
            seq_file,
            seq_args_string,
            num_reps,
        )

    # ---------------------------------------------------------
    # Configure MW sources for every acquisition step
    # ---------------------------------------------------------
    def step_fn(step_ind):
        # NV source: fixed ESR frequency/power from virtual sig-gen config.
        nv_dict = tb.get_virtual_sig_gen_dict(
            nv_ind
        )
        nv_mw = tb.get_server_sig_gen(
            nv_ind
        )

        nv_mw.set_amp(
            nv_dict["uwave_power"]
        )
        nv_mw.set_freq(
            nv_dict["frequency"]
        )
        nv_mw.uwave_on()

        # P1 source: ON or detuned OFF frequency.
        rf_dict = tb.get_virtual_sig_gen_dict(
            rf_ind
        )
        rf = tb.get_server_sig_gen(
            rf_ind
        )

        rf.set_amp(
            rf_dict["uwave_power"]
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
    # Acquire
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

    # ---------------------------------------------------------
    # Process ON/OFF pairs
    # ---------------------------------------------------------
    raw_fig = None

    try:
        counts = np.asarray(
            raw_data["counts"]
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
            sig_counts, ref_counts = (
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
            contrast_centered,
            p25,
            p50,
            p75,
        ) = create_deer_rabi_figure(
            rf_len_ns,
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
            "contrast_centered": np.ascontiguousarray(
                contrast_centered
            ),
            "median_contrast": np.ascontiguousarray(
                p50
            ),
            "p25_contrast": np.ascontiguousarray(
                p25
            ),
            "p75_contrast": np.ascontiguousarray(
                p75
            ),
        }

    except Exception:
        print(
            traceback.format_exc()
        )

    # ---------------------------------------------------------
    # Reset hardware
    # ---------------------------------------------------------
    tb.reset_cfm()
    kpl.show()

    # ---------------------------------------------------------
    # Save using the same DM workflow as deer_hahn.py
    # ---------------------------------------------------------
    timestamp = dm.get_time_stamp()

    raw_data |= {
        "timestamp": timestamp,
        "experiment": "deer_hahn_rabi",

        # Physical Rabi axis
        "rf_len_ns": np.ascontiguousarray(
            rf_len_ns
        ),
        "rf_len_units": "ns",

        # Full sequence-step arrays
        "rf_len_ns_interleaved": np.ascontiguousarray(
            rf_len_interleaved
        ),
        "rf_freq_ghz": float(
            rf_freq_ghz
        ),
        "rf_freq_off_ghz": float(
            rf_freq_off_ghz
        ),
        "rf_freq_interleaved_ghz": np.ascontiguousarray(
            rf_freq_interleaved
        ),

        # DEER configuration
        "ref_detuning_ghz": float(
            ref_detuning_ghz
        ),
        "tau_ns": int(
            tau_ns
        ),
        "total_free_evolution_ns": int(
            2 * tau_ns
        ),
        "nv_pi_ns": int(
            nv_pi_ns
        ),
        "center_rf_on_nv_pi": bool(
            center_rf_on_nv_pi
        ),
        "dynamic_thresh": bool(
            dynamic_thresh
        ),

        # Hardware roles
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
    repr_nv_name = repr_nv_sig.name

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_name,
    )

    keys_to_compress = [
        "rf_len_ns",
        "rf_len_ns_interleaved",
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
        "contrast_centered",
        "median_contrast",
        "p25_contrast",
        "p75_contrast",
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

    # Example call belongs in your experiment launcher once nv_list is loaded.
    #
    # Suggested initial P1 Rabi sweep around the current ~100 ns pi pulse:
    #
    # main(
    #     nv_list=nv_list,
    #     num_steps=51,
    #     num_reps=2,
    #     num_runs=100,
    #     min_rf_len_ns=20,
    #     max_rf_len_ns=1000,
    #     rf_freq_ghz=0.200,   # replace with selected JT resonance
    #     uwave_ind_list=[0, 1],
    #     tau_ns=18_000,
    #     nv_pi_ns=256,
    #     ref_detuning_ghz=0.6,
    # )
    #
    plt.show(block=True)
