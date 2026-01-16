# # -*- coding: utf-8 -*-
# """
# Widefield ESR

# Created on October 3th, 2025

# @author: schand
# """

# import time

# import matplotlib.pyplot as plt
# import numpy as np
# from qm import QuantumMachinesManager, qua
# from qm.simulate import SimulationConfig

# import utils.common as common
# from servers.timing.sequencelibrary.QM_opx import seq_utils
# from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


# def get_seq(
#     base_scc_seq_args,
#     step_inds=None,
#     num_reps=1,
# ):
#     reference = False  # References for this sequence are handled routine-side
#     buffer = seq_utils.get_widefield_operation_buffer()
#     # revival = 19.6e4 # in ns
#     revival = 15e3 # in ns
#     step_val = seq_utils.convert_ns_to_cc(revival)
#     with qua.program() as seq:
#         seq_utils.init()
#         seq_utils.macro_run_aods()
#         step_ind = qua.declare(int)

#         def uwave_macro(uwave_ind_list, step_ind):
#             MW_NV = [uwave_ind_list[0],uwave_ind_list[1]]  # NV microwave chai
#             RF = [uwave_ind_list[2]]  # RF chain
#             qua.align()
#             seq_utils.macro_pi_on_2_pulse(MW_NV)
#             qua.wait(step_val)
#             seq_utils.macro_pi_pulse(MW_NV)
#             seq_utils.macro_pi_pulse(RF)
#             qua.wait(step_val)
#             seq_utils.macro_pi_on_2_pulse(MW_NV)
#             qua.wait(buffer)

#         with qua.for_each_(step_ind, step_inds):
#             base_scc_sequence.macro(
#                 base_scc_seq_args,
#                 uwave_macro,
#                 step_ind,
#                 num_reps=num_reps,
#                 reference=reference,
#             )

#     seq_ret_vals = []
#     return seq, seq_ret_vals


# if __name__ == "__main__":
#     config_module = common.get_config_module()
#     config = config_module.config
#     opx_config = config_module.opx_config
#     opx_config["pulses"]["yellow_spin_pol"]["length"] = 1e3

#     qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
#     qmm = QuantumMachinesManager(**qm_opx_args)
#     opx = qmm.open_qm(opx_config)

#     try:
#         seq, seq_ret_vals = get_seq(
#             [
#                 [[108.477, 107.282], [109.356, 108.789]],
#                 [220, 220],
#                 [1.0, 1.0],
#                 [[73.558, 71.684], [74.227, 72.947]],
#                 [124, 124],
#                 [1.0, 1.0],
#                 [False, False],
#                 [0,1,2],
#             ],
#             [70, 219],
#             1,
#         )

#         sim_config = SimulationConfig(duration=int(300e3 / 4))
#         sim = opx.simulate(seq, sim_config)
#         samples = sim.get_simulated_samples()
#         samples.con1.plot()
#         plt.show(block=True)

#     except Exception as exc:
#         raise exc
#     finally:
#         qmm.close_all_quantum_machines()


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


# -----------------------------
# IMPORTANT: index -> element mapping
# -----------------------------
# You used uwave_ind_list = [0, 1, 2] where:
#   0,1 = NV microwave chains
#   2   = RF chain (gated by DO on con1,3 in your config)
#
# In your posted config, those digital modulation elements are:
#   do_sig_gen_STAN_sg394_0_dm  -> do_pi_pulse_0, do_pi_on_2_pulse_0
#   do_sig_gen_STAN_sg394_1_dm  -> do_pi_pulse_1, do_pi_on_2_pulse_1
#   do_sig_gen_STAN_sg394_3_dm  -> do_pi_pulse_2, do_pi_on_2_pulse_2
#
# If your local naming differs, update ONLY this dict.
UWAVE_DO_ELEM_BY_IND = {
    0: "do_sig_gen_STAN_sg394_0_dm",
    1: "do_sig_gen_STAN_sg394_1_dm",
    2: "do_sig_gen_STAN_sg394_3_dm",  # (yes, "_3_" but pulses are "_2")
}


def _ns_to_cc(ns: int) -> int:
    """OPX clock is 4 ns. Quantize ns -> cc with rounding to nearest 4 ns."""
    ns_q = int(4 * round(ns / 4))
    return seq_utils.convert_ns_to_cc(ns_q)


def get_seq(
    base_scc_seq_args,
    step_inds=None,
    
    num_reps=1,
):
    reference = False  # references handled routine-side
    buffer_cc = seq_utils.get_widefield_operation_buffer()
    # rf_pi_ns = seq_utils.get_common_duration_cc
    # -------- user knobs --------
    tau_ns = 18_600          # 15 us (C13 revival target you mentioned)
    nv_pi_ns = 100           # your NV pi pulse length (digital gate)
    rf_pi_ns = 2_000         # your RF pi pulse length (digital gate)
    # ----------------------------

    tau_cc = _ns_to_cc(tau_ns)

    # Center RF pulse on NV π pulse:
    # RF starts earlier by delta = (Lrf - Lnv)/2.
    delta_ns = (rf_pi_ns - nv_pi_ns) // 2  # 950 ns for (2000-100)/2
    delta_cc = _ns_to_cc(delta_ns)

    # RF should start at (tau - delta) relative to the time right after NV pi/2 completes
    pre_cc = tau_cc - delta_cc
    if pre_cc < 0:
        raise ValueError(f"tau is too short to center RF: tau_cc={tau_cc}, delta_cc={delta_cc}")

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        step_ind = qua.declare(int)

        def uwave_macro(uwave_ind_list, step_ind):
            # indices
            nv_inds = [uwave_ind_list[0], uwave_ind_list[1]]
            rf_ind = [uwave_ind_list[2]]

            # element names
            nv_elems = [UWAVE_DO_ELEM_BY_IND[i] for i in nv_inds]
            rf_elem = UWAVE_DO_ELEM_BY_IND[rf_ind[0]]

            # 1) Start all 3 timelines together
            qua.align(*nv_elems, rf_elem)

            # 2) NV pi/2 on BOTH NV chains (same start time)
            # for e in nv_elems:
            #     qua.play("pi_on_2_pulse", e)
            seq_utils.macro_pi_on_2_pulse(nv_inds)
                
            
            # Align again so "tau" starts from the *end* of the pi/2 pulses
            qua.align(*nv_elems, rf_elem)

            # 3) Schedule RF so its midpoint coincides with the NV π midpoint
            # RF: wait (tau - delta) then play RF pi pulse (2 us)
            qua.wait(pre_cc, rf_elem)
            seq_utils.macro_pi_pulse(rf_ind)

            # 4) NV π occurs at exactly tau after the post-pi/2 alignment
            for e in nv_elems:
                qua.wait(tau_cc, e)
            for e in nv_elems:
                seq_utils.macro_pi_pulse(nv_inds)
                

            # 5) second free evolution tau, then final NV pi/2
            for e in nv_elems:
                qua.wait(tau_cc, e)
            for e in nv_elems:
                seq_utils.macro_pi_on_2_pulse(nv_inds)
                

            # 6) bring everything back together before leaving the macro
            qua.align(*nv_elems, rf_elem)
            qua.wait(buffer_cc)

        with qua.for_each_(step_ind, step_inds):
            base_scc_sequence.macro(
                base_scc_seq_args,
                uwave_macro,
                step_ind,
                num_reps=num_reps,
                reference=reference,
            )

    seq_ret_vals = []
    return seq, seq_ret_vals


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
                [0, 1, 2],
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
