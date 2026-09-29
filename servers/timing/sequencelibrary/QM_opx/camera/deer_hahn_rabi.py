# -*- coding: utf-8 -*-
"""
Widefield DEER-Hahn Rabi sequence.

Sweep the P1/RF pulse duration while keeping the NV Hahn echo fixed.

Timing:
    NV:  pi/2 -- tau -- pi -- tau -- pi/2

For each Rabi point, the P1 pulse is centered on the NV pi pulse:

    t_RF,start = tau + (t_pi,NV - t_RF) / 2

Channel convention (same as deer_hahn.py):
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


def _ns_to_cc(ns: float) -> int:
    """Quantize ns to the 4 ns OPX clock and convert to clock cycles."""
    ns_q = int(4 * round(float(ns) / 4))
    return seq_utils.convert_ns_to_cc(ns_q)


def get_seq(
    base_scc_seq_args,
    rf_len_ns_list,
    tau_ns=18_000,
    nv_pi_ns=256,
    center_rf_on_nv_pi=True,
    num_reps=1,
):
    """
    Build the DEER-Hahn Rabi sequence.

    Parameters
    ----------
    base_scc_seq_args
        Standard SCC sequence arguments. The final item must contain
        [NV_ind, RF_ind].
    rf_len_ns_list
        P1/RF pulse durations for the sequence steps, in ns.
        The host routine may repeat each duration twice for ON/OFF
        frequency referencing.
    tau_ns
        Hahn tau in ns. Total free evolution is 2*tau.
    nv_pi_ns
        Actual NV pi-pulse duration used for centering.
    center_rf_on_nv_pi
        If True, center the variable P1 pulse on the NV pi pulse.
    num_reps
        Repetitions supplied by stream_load().
    """
    reference = False
    buffer_cc = seq_utils.get_widefield_operation_buffer()

    if len(rf_len_ns_list) == 0:
        raise ValueError("rf_len_ns_list cannot be empty.")

    rf_len_ns_list = [
        int(4 * round(float(val) / 4))
        for val in rf_len_ns_list
    ]

    if any(val <= 0 for val in rf_len_ns_list):
        raise ValueError("All P1/RF pulse durations must be > 0 ns.")

    tau_cc = _ns_to_cc(tau_ns)
    nv_pi_cc = _ns_to_cc(nv_pi_ns)
    rf_len_cc_list = [
        _ns_to_cc(val)
        for val in rf_len_ns_list
    ]

    if center_rf_on_nv_pi:
        max_rf_cc = 2 * tau_cc + nv_pi_cc

        for rf_cc, rf_ns in zip(
            rf_len_cc_list,
            rf_len_ns_list,
        ):
            if rf_cc > max_rf_cc:
                raise ValueError(
                    "P1/RF pulse is too long to remain centered on the "
                    "NV pi pulse: "
                    f"{rf_ns} ns > approximately "
                    f"{2*tau_ns + nv_pi_ns} ns."
                )

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        rf_len_cc = qua.declare(int)

        def uwave_macro(uwave_ind_list, rf_len_cc):
            if len(uwave_ind_list) != 2:
                raise ValueError(
                    "DEER-Hahn Rabi expects exactly "
                    "[NV_ind, RF_ind]."
                )

            nv_ind = uwave_ind_list[0]
            rf_ind = uwave_ind_list[1]

            nv_elem = UWAVE_DO_ELEM_BY_IND[nv_ind]
            rf_elem = UWAVE_DO_ELEM_BY_IND[rf_ind]

            # Start together.
            qua.align(nv_elem, rf_elem)

            # NV pi/2.
            seq_utils.macro_pi_on_2_pulse([nv_ind])

            # Define t=0 after the first pi/2.
            qua.align(nv_elem, rf_elem)

            # P1/RF pulse.
            if center_rf_on_nv_pi:
                # Center P1 pulse on the NV pi pulse:
                # tau + (NVpi - RFlen)/2.
                rf_start_cc = (
                    tau_cc
                    + ((nv_pi_cc - rf_len_cc) >> 1)
                )
            else:
                rf_start_cc = tau_cc

            qua.wait(rf_start_cc, rf_elem)
            seq_utils.macro_pi_pulse(
                [rf_ind],
                duration_cc=rf_len_cc,
            )

            # Central NV pi at exactly tau.
            qua.wait(tau_cc, nv_elem)
            seq_utils.macro_pi_pulse([nv_ind])

            # Second Hahn arm.
            qua.wait(tau_cc, nv_elem)
            seq_utils.macro_pi_on_2_pulse([nv_ind])

            # Finish only after both timelines are complete.
            qua.align(nv_elem, rf_elem)
            qua.wait(
                buffer_cc,
                nv_elem,
                rf_elem,
            )

        with qua.for_each_(
            rf_len_cc,
            rf_len_cc_list,
        ):
            base_scc_sequence.macro(
                base_scc_seq_args,
                uwave_macro,
                rf_len_cc,
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
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        # Example only: around a ~100 ns P1 pi pulse.
        rf_len_ns_list = list(range(20, 1001, 20))

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
            rf_len_ns_list,
            tau_ns=18_000,
            nv_pi_ns=256,
            center_rf_on_nv_pi=True,
            num_reps=1,
        )

        sim_config = SimulationConfig(
            duration=int(300e3 / 4)
        )
        sim = opx.simulate(
            seq,
            sim_config,
        )
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    finally:
        qmm.close_all_quantum_machines()
