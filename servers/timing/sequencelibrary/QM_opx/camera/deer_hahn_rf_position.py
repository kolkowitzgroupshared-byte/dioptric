# -*- coding: utf-8 -*-
"""
Widefield DEER-Hahn RF-position sweep.

Fixed:
    P1/RF frequency = configured by host (e.g. 198 MHz)
    P1/RF pulse length = fixed (default 400 ns)
    NV Hahn echo = pi/2 -- tau -- pi -- tau -- pi/2

Swept:
    Center time of the P1/RF pulse relative to the center of the NV pi pulse.

Definition:
    rf_center_offset_ns = 0
        -> P1/RF pulse center aligned with NV-pi center

    rf_center_offset_ns < 0
        -> P1/RF pulse occurs before NV pi

    rf_center_offset_ns > 0
        -> P1/RF pulse occurs after NV pi

Channel convention:
    uwave_ind_list[0] = selected NV microwave source
    uwave_ind_list[1] = P1 / RF source

All programmed times are quantized to the OPX 4 ns clock.

@author: schand
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


UWAVE_DO_ELEM_BY_IND = {
    0: "do_sig_gen_STAN_sg394_0_dm",
    1: "do_sig_gen_STAN_sg394_1_dm",
}


def _quantize_4ns(ns: float) -> int:
    return int(4 * round(float(ns) / 4))


def _ns_to_cc(ns: float) -> int:
    ns_q = _quantize_4ns(ns)
    return seq_utils.convert_ns_to_cc(ns_q)


def get_seq(
    base_scc_seq_args,
    rf_center_offset_ns_list,
    rf_len_ns=400,
    tau_ns=18_000,
    nv_pi_ns=256,
    num_reps=1,
):
    """
    Build the DEER-Hahn RF-position sweep.

    Parameters
    ----------
    base_scc_seq_args
        Standard SCC sequence args. Final microwave list must be
        [NV_ind, RF_ind].
    rf_center_offset_ns_list
        P1/RF pulse-center offsets relative to the NV-pi center.
        Negative = before NV pi, positive = after NV pi.
    rf_len_ns
        Fixed P1/RF pulse duration.
    tau_ns
        Hahn tau. Total free-evolution time is 2*tau.
    nv_pi_ns
        NV pi pulse duration.
    num_reps
        Repetitions supplied by stream_load().
    """
    reference = False
    buffer_cc = seq_utils.get_widefield_operation_buffer()

    if len(rf_center_offset_ns_list) == 0:
        raise ValueError(
            "rf_center_offset_ns_list cannot be empty."
        )

    rf_len_ns = _quantize_4ns(rf_len_ns)
    tau_ns = _quantize_4ns(tau_ns)
    nv_pi_ns = _quantize_4ns(nv_pi_ns)

    if rf_len_ns <= 0:
        raise ValueError("rf_len_ns must be > 0.")

    # t=0 is defined after the first NV pi/2.
    #
    # NV pi:
    #   starts at tau
    #   center = tau + nv_pi/2
    #
    # RF pulse center:
    #   NV-pi center + offset
    #
    # RF start:
    #   tau + nv_pi/2 - rf_len/2 + offset
    #
    nominal_rf_start_ns = (
        tau_ns
        + (nv_pi_ns - rf_len_ns) / 2
    )

    sequence_end_ns = (
        2 * tau_ns
        + nv_pi_ns
    )

    rf_start_ns_list = []

    for offset_ns in rf_center_offset_ns_list:
        offset_ns_q = _quantize_4ns(offset_ns)
        rf_start_ns = _quantize_4ns(
            nominal_rf_start_ns
            + offset_ns_q
        )

        rf_end_ns = (
            rf_start_ns
            + rf_len_ns
        )

        if rf_start_ns < 0:
            raise ValueError(
                "RF pulse starts before the first Hahn arm. "
                f"offset={offset_ns_q} ns gives "
                f"rf_start={rf_start_ns} ns."
            )

        if rf_end_ns > sequence_end_ns:
            raise ValueError(
                "RF pulse ends after the second Hahn arm. "
                f"offset={offset_ns_q} ns gives "
                f"rf_end={rf_end_ns} ns, "
                f"sequence_end={sequence_end_ns} ns."
            )

        rf_start_ns_list.append(
            rf_start_ns
        )

    tau_cc = _ns_to_cc(tau_ns)
    rf_len_cc = _ns_to_cc(rf_len_ns)

    rf_start_cc_list = [
        _ns_to_cc(val)
        for val in rf_start_ns_list
    ]

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        rf_start_cc = qua.declare(int)

        def uwave_macro(
            uwave_ind_list,
            rf_start_cc,
        ):
            if len(uwave_ind_list) != 2:
                raise ValueError(
                    "DEER-Hahn RF-position sweep expects "
                    "exactly [NV_ind, RF_ind]."
                )

            nv_ind = uwave_ind_list[0]
            rf_ind = uwave_ind_list[1]

            nv_elem = UWAVE_DO_ELEM_BY_IND[
                nv_ind
            ]
            rf_elem = UWAVE_DO_ELEM_BY_IND[
                rf_ind
            ]

            qua.align(
                nv_elem,
                rf_elem,
            )

            # First NV pi/2.
            seq_utils.macro_pi_on_2_pulse(
                [nv_ind]
            )

            # t=0 after first pi/2.
            qua.align(
                nv_elem,
                rf_elem,
            )

            # Fixed-length P1 pulse at swept position.
            qua.wait(
                rf_start_cc,
                rf_elem,
            )
            seq_utils.macro_pi_pulse(
                [rf_ind],
                duration_cc=rf_len_cc,
            )

            # Central NV pi remains fixed.
            qua.wait(
                tau_cc,
                nv_elem,
            )
            seq_utils.macro_pi_pulse(
                [nv_ind]
            )

            # Second Hahn arm.
            qua.wait(
                tau_cc,
                nv_elem,
            )
            seq_utils.macro_pi_on_2_pulse(
                [nv_ind]
            )

            # Wait for both element timelines.
            qua.align(
                nv_elem,
                rf_elem,
            )
            qua.wait(
                buffer_cc,
                nv_elem,
                rf_elem,
            )

        with qua.for_each_(
            rf_start_cc,
            rf_start_cc_list,
        ):
            base_scc_sequence.macro(
                base_scc_seq_args,
                uwave_macro,
                rf_start_cc,
                num_reps=num_reps,
                reference=reference,
            )

    return seq, []


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    opx_config["pulses"]["yellow_spin_pol"]["length"] = 1e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(
        **qm_opx_args
    )
    opx = qmm.open_qm(
        opx_config
    )

    try:
        # Example:
        # 400 ns P1 pulse, sweep center over almost full Hahn window.

        seq, _ = get_seq(
            [
                [[108.477, 107.282], [109.356, 108.789]],
                [220, 220],
                [1.0, 1.0],
                [[73.558, 71.684], [74.227, 72.947]],
                [124, 124],
                [1.0, 1.0],
                [False, False],
                [0, 1],
            ],
            [-5000, 5000], ##
            rf_len_ns=400,
            tau_ns=18_000,
            nv_pi_ns=256,
            num_reps=1,
        )

        sim_config = SimulationConfig(
            duration=int(200e3 / 4)
        )
        sim = opx.simulate(
            seq,
            sim_config,
        )
        samples = (
            sim.get_simulated_samples()
        )
        samples.con1.plot()
        plt.show(block=True)

    finally:
        qmm.close_all_quantum_machines()
