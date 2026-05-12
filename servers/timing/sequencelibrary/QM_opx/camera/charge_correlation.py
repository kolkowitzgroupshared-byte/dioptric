# -*- coding: utf-8 -*-
"""
Charge correlation sequence wrapper.

Per repetition:
    optional charge drive / polarization
    charge-state readout
    wait for trigger
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_charge_correlation


def get_seq(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    do_drive,
    targeted_drive,
    num_reps,
):
    with seq_utils.qua.program() as seq:
        num_nvs = len(pol_coords_list)
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        base_charge_correlation.macro(
            pol_coords_list=pol_coords_list,
            pol_duration_list=pol_duration_list,
            pol_amp_list=pol_amp_list,
            do_drive=do_drive,
            targeted_drive=targeted_drive,
            num_reps=num_reps,
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