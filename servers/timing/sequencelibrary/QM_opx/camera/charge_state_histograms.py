# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

Created on October 13th, 2023

@author: mccambria
@author: schand
"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_charge_state_histograms


def get_seq(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    ion_coords_list,
    ion_duration_list,
    ion_amp_list,
    ion_do_target_list,
    verify_charge_states,
    num_reps,
):
    with qua.program() as seq:
        num_nvs = len(pol_coords_list)

        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        base_charge_state_histograms.macro(
            pol_coords_list=pol_coords_list,
            pol_duration_list=pol_duration_list,
            pol_amp_list=pol_amp_list,
            ion_coords_list=ion_coords_list,
            ion_duration_list=ion_duration_list,
            ion_amp_list=ion_amp_list,
            num_reps=num_reps,
            ion_do_target_list=ion_do_target_list,
            verify_charge_states=verify_charge_states,
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
            # pol_coords_list
            [[109.033, 106.685], [115.694, 101.182]],

            # pol_duration_list, ns
            [10000, 10000],

            # pol_amp_list, QUA multipliers
            [1.0, 1.0],

            # ion_coords_list
            [[73.524, 72.417], [78.906, 67.807]],

            # ion_duration_list, ns
            [10000, 10000],

            # ion_amp_list, QUA multipliers
            [1.0, 1.0],

            # ion_do_target_list
            None,

            # verify_charge_states
            False,

            # num_reps
            5,
        )

        sim_config = SimulationConfig(duration=int(150e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()

    except Exception as exc:
        raise exc

    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)