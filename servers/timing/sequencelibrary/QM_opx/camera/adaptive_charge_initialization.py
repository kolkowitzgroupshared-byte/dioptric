# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation.

Used for both:
    1. old conditional initialization
    2. DMD conditional initialization

DMD is NOT controlled here.
DMD is controlled in Python charge_prep_fn.
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(
    ion_coords_list,
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    num_reps,
):
    if num_reps is None:
        num_reps = 1

    num_nvs = len(pol_coords_list)

    with qua.program() as seq:
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        def one_rep(rep_ind=None):
            # rep 0: ionize all selected NVs, then readout
            with qua.if_(rep_ind == 0):
                seq_utils.macro_ionize(ion_coords_list)

            # rep > 0: conditional charge polarization, then readout
            with qua.else_():
                seq_utils.macro_polarize(
                    pol_coords_list,
                    pol_duration_list,
                    pol_amp_list,
                    spin_pol=False,
                    targeted_polarization=True,
                    verify_charge_states=False,
                )

            seq_utils.macro_charge_state_readout()
            seq_utils.macro_wait_for_trigger()

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        seq_utils.macro_pause()

    return seq, []


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            ion_coords_list=[[110, 110], [112, 112]],
            pol_coords_list=[[110, 110], [112, 112]],
            pol_duration_list=[1000, 1000],
            pol_amp_list=[0.1, 0.1],
            do_charge_polarize=True,
            num_reps=3,
        )

        sim_config = SimulationConfig(duration=int(150e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()

    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)