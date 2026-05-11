# -*- coding: utf-8 -*-
"""
Simple DMD crosstalk readout sequence.

This sequence does not control the DMD. The Python experiment sets a static DMD
mask before each acquisition.

Per repetition:
    optional charge polarization
    widefield charge-state readout
    wait for trigger

Put this file in:
    servers/timing/sequencelibrary/QM_opx/camera/dmd_crosstalk_readout.py
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    do_polarize,
    targeted_polarization,
    num_reps,
):
    if num_reps is None:
        num_reps = 1

    num_nvs = len(pol_coords_list)

    with qua.program() as seq:
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        def one_rep(rep_ind=None):
            if do_polarize:
                seq_utils.macro_polarize(
                    pol_coords_list,
                    pol_duration_list,
                    pol_amp_list,
                    targeted_polarization=targeted_polarization,
                    verify_charge_states=False,
                    spin_pol=False,
                )

            seq_utils.macro_charge_state_readout()
            seq_utils.macro_wait_for_trigger()

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
        seq_utils.macro_pause()

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
        seq, seq_ret_vals = get_seq(
            [[109.033, 106.685], [115.694, 101.182]],
            [10000, 10000],
            [1.0, 1.0],
            True,
            False,
            5,
        )

        sim_config = SimulationConfig(duration=int(150e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()

    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)
