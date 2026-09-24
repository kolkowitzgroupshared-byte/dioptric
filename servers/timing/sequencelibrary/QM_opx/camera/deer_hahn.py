# -*- coding: utf-8 -*-
"""
Widefield DEER-style echo (widefield SCC wrapper)

Goal here: make the RF π pulse (2 us) centered on the NV π pulse (100 ns),
i.e. the NV π pulse sits symmetrically inside the RF pulse.

This version does the centering *in the QUA schedule* using element-specific waits,
so you don’t need any negative config delays.

Created on October 3th, 2025
@author: schand
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

# If your local naming differs, update ONLY this dict.
UWAVE_DO_ELEM_BY_IND = {
    0: "do_sig_gen_STAN_sg394_0_dm",
    1: "do_sig_gen_STAN_sg394_1_dm",
}


def _ns_to_cc(ns: int) -> int:
    """OPX clock is 4 ns. Quantize ns -> cc with rounding to nearest 4 ns."""
    ns_q = int(4 * round(ns / 4))
    return seq_utils.convert_ns_to_cc(ns_q)


def get_seq(
    base_scc_seq_args,
    step_inds=None,
    tau_ns=18_000,
    nv_pi_ns=256,
    rf_pi_ns=100,
    num_reps=1,
):
    reference = False
    buffer_cc = seq_utils.get_widefield_operation_buffer()

    tau_cc = _ns_to_cc(tau_ns)

    # Center RF/P1 pi inside NV pi.
    rf_start_ns = tau_ns + (nv_pi_ns - rf_pi_ns) / 2
    rf_start_cc = _ns_to_cc(rf_start_ns)

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_ind = qua.declare(int)

        def uwave_macro(uwave_ind_list, step_ind):

            # 0 = NV
            # 1 = P1 / RF
            nv_ind = uwave_ind_list[0]
            rf_ind = uwave_ind_list[1]

            nv_elem = UWAVE_DO_ELEM_BY_IND[nv_ind]
            rf_elem = UWAVE_DO_ELEM_BY_IND[rf_ind]

            qua.align(nv_elem, rf_elem)

            # NV pi/2
            seq_utils.macro_pi_on_2_pulse([nv_ind])

            # Define t=0 after first pi/2
            qua.align(nv_elem, rf_elem)

            # P1/RF pi centered within NV pi
            qua.wait(rf_start_cc, rf_elem)
            seq_utils.macro_pi_pulse([rf_ind])

            # Central NV pi
            qua.wait(tau_cc, nv_elem)
            seq_utils.macro_pi_pulse([nv_ind])

            # Second Hahn arm
            qua.wait(tau_cc, nv_elem)
            seq_utils.macro_pi_on_2_pulse([nv_ind])

            qua.align(nv_elem, rf_elem)
            qua.wait(buffer_cc, nv_elem, rf_elem)

        with qua.for_each_(step_ind, step_inds):
            base_scc_sequence.macro(
                base_scc_seq_args,
                uwave_macro,
                step_ind,
                num_reps=num_reps,
                reference=reference,
            )

    return seq, []


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    # example tweak
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 1e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
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
            step_inds=[70, 219],
            num_reps=1,
        )

        sim_config = SimulationConfig(duration=int(300e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    finally:
        qmm.close_all_quantum_machines()
