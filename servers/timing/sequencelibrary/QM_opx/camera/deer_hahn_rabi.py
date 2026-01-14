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
# import utils.tool_belt as tb

# import utils.common as common
# from servers.timing.sequencelibrary.QM_opx import seq_utils
# from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


# def get_seq(
#     base_scc_seq_args,
#     step_vals,
#     num_reps=1,
# ):
#     buffer = seq_utils.get_widefield_operation_buffer()
#     step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals]
#     revival = 15e3 # in ns
#     revival_val = seq_utils.convert_ns_to_cc(revival)
#     with qua.program() as seq:
#         seq_utils.init()
#         seq_utils.macro_run_aods()
#         step_val = qua.declare(int)
#         def uwave_macro(uwave_ind_list, step_val):
#             MW_NV = [uwave_ind_list[1]]  # NV microwave chain (~2.87 GHz)
#             RF = [uwave_ind_list[0]]  # RF chain (~133 MHz)
#             qua.align()
#             seq_utils.macro_pi_on_2_pulse(MW_NV, phase=0)
#             qua.wait(revival_val)
#             seq_utils.macro_pi_pulse(RF,  duration_cc=step_val, phase=0)
#             seq_utils.macro_pi_pulse(MW_NV, phase=0)
#             qua.wait(revival_val)
#             seq_utils.macro_pi_on_2_pulse(MW_NV, phase=0)
#             qua.wait(buffer)

#         with qua.for_each_(step_val, step_vals):
#             base_scc_sequence.macro(
#                 base_scc_seq_args,
#                 uwave_macro,
#                 step_val,
#                 num_reps=num_reps,
#             )

#     seq_ret_vals = []
#     return seq, seq_ret_vals


# if __name__ == "__main__":
#     config_module = common.get_config_module()
#     config = config_module.config
#     opx_config = config_module.opx_config
#     opx_config["pulses"]["yellow_spin_pol"]["length"] = 1e3
#     tb.set_delays_to_zero(opx_config)

#     qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
#     qmm = QuantumMachinesManager(**qm_opx_args)
#     opx = qmm.open_qm(opx_config)

#     try:
#         seq, seq_ret_vals = get_seq(
#             [
#                 [
#                     [107.247, 107.388],
#                     [95.571, 94.737],
#                 ],
#                 [188, 108],
#                 [1.0, 1.0],
#                 [
#                     [72.0, 72.997],
#                     [62.207, 62.846],
#                 ],
#                 [104, 56],
#                 [1.0, 1.0],
#                 [False, False],
#                 [0,1],
#             ],
#             [
#                 112.0,
#                 184.0,
#             ],
#             1,
#         )

#         sim_config = SimulationConfig(duration=int(200e3/4))
#         sim = opx.simulate(seq, sim_config)
#         samples = sim.get_simulated_samples()
#         samples.con1.plot()
#         plt.show(block=True)

#     except Exception as exc:
#         print(f"An error occurred: {exc}")
#     finally:
#         qmm.close_all_quantum_machines()


# -*- coding: utf-8 -*-
"""
Widefield DEER-style echo with RF Rabi (sweep RF pulse duration)

- Keep RF frequency fixed in the sig gen (sit on one peak)
- Sweep RF pulse length to see Rabi oscillations in the DEER contrast
- Centers RF pulse on NV pi pulse (optional but default True)

@author: schand
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


# Map uwave index -> DO element name (same as your working freq sweep)
UWAVE_DO_ELEM_BY_IND = {
    0: "do_sig_gen_STAN_sg394_0_dm",
    1: "do_sig_gen_STAN_sg394_1_dm",
    2: "do_sig_gen_STAN_sg394_3_dm",  # RF chain
}


def _ns_to_cc(ns: float) -> int:
    """Quantize ns to 4 ns and convert to OPX clock cycles."""
    ns_q = int(4 * round(ns / 4))
    return seq_utils.convert_ns_to_cc(ns_q)


def get_seq(
    base_scc_seq_args,
    rf_len_ns_list,            # <-- sweep these (ns)
    num_reps=1,
    tau_ns=15_000,             # echo tau (ns)
    nv_pi_ns=100,              # NV pi length for centering (ns) (digital gate length)
    center_rf_on_nv_pi=True,   # True = your "centered RF" style
):
    base_scc_seq_args[-1] = [0, 1] # NV-only (example)
    reference = True
    buffer_cc = seq_utils.get_widefield_operation_buffer()

    tau_cc = _ns_to_cc(tau_ns)
    nv_pi_cc = _ns_to_cc(nv_pi_ns)

    # Precompute RF pulse durations (cc) on host
    rf_len_cc_list = [_ns_to_cc(x) for x in rf_len_ns_list]
    max_rf_cc = max(rf_len_cc_list)

    # Basic feasibility checks for centering
    if center_rf_on_nv_pi:
        for rf_cc in rf_len_cc_list:
            # pre_cc = tau - (rf-nv)/2 must be >= 0  -> rf <= 2*tau + nv
            if rf_cc > (2 * tau_cc + nv_pi_cc):
                raise ValueError(
                    f"RF pulse too long for centering: rf_len ~ {rf_cc}cc exceeds 2*tau+nv_pi."
                )
            if rf_cc < nv_pi_cc:
                raise ValueError(
                    f"RF pulse shorter than NV pi ({nv_pi_ns} ns) cannot be centered on NV pi."
                )

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        rf_len_cc = qua.declare(int)

        def uwave_macro(uwave_ind_list, rf_len_cc):
            # Expect uwave_ind_list = [nv0, nv1, rf]
            nv_inds = [uwave_ind_list[0], uwave_ind_list[1]]
            rf_ind = [2]

            nv_elems = [UWAVE_DO_ELEM_BY_IND[i] for i in nv_inds]
            rf_elem = UWAVE_DO_ELEM_BY_IND[rf_ind[0]]

            # 1) Start together
            qua.align(*nv_elems, rf_elem)

            # 2) NV pi/2
            seq_utils.macro_pi_on_2_pulse(nv_inds)

            # Start "tau" from end of pi/2 pulses
            qua.align(*nv_elems, rf_elem)

            # 3) Schedule RF 
            if center_rf_on_nv_pi:
                # delta = (Lrf - Lnv)/2  (integer)
                delta_cc = (rf_len_cc - nv_pi_cc) >> 1
                pre_cc = tau_cc - delta_cc
            else:
                pre_cc = tau_cc


            qua.wait(pre_cc, rf_elem)
            seq_utils.macro_pi_pulse(rf_ind, duration_cc=rf_len_cc)

            # Optional: pad RF element so total macro length is constant vs rf_len
            # (helps keep heating/duty-cycle consistent)
            rf_cc = max_rf_cc - rf_len_cc
            qua.wait(rf_cc, rf_elem)

            # 4) NV pi at exactly tau after the pi/2
            for e in nv_elems:
                qua.wait(tau_cc, e)
            seq_utils.macro_pi_pulse(nv_inds)

            # 5) second tau, then final NV pi/2
            for e in nv_elems:
                qua.wait(tau_cc, e)
            seq_utils.macro_pi_on_2_pulse(nv_inds)

            # 6) buffer and exit
            qua.align(*nv_elems, rf_elem)
            qua.wait(buffer_cc)

        with qua.for_each_(rf_len_cc, rf_len_cc_list):
            base_scc_sequence.macro(
                base_scc_seq_args,
                uwave_macro,
                rf_len_cc,
                num_reps=num_reps,
                reference=reference,
            )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        # Example: sweep RF pulse length 0.2–8 us
        rf_len_ns_list = [200, 400, 800, 1200, 1600, 2000, 2600, 3200, 4000, 6000, 8000]

        seq, _ = get_seq(
            base_scc_seq_args=[
                [[108.477, 107.282], [109.356, 108.789]],
                [220, 220],
                [1.0, 1.0],
                [[73.558, 71.684], [74.227, 72.947]],
                [124, 124],
                [1.0, 1.0],
                [False, False],
                [0, 1, 2],  # NV0, NV1, RF
            ],
            rf_len_ns_list=rf_len_ns_list,
            num_reps=1,
            tau_ns=15_000,
            nv_pi_ns=100,
            center_rf_on_nv_pi=True,
        )

        sim_config = SimulationConfig(duration=int(350e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    finally:
        qmm.close_all_quantum_machines()
