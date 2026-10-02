# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

Created on October 13th, 2023

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(
    ion_coords_list, pol_coords_list, pol_duration_list, pol_amp_list, num_reps
):
    if num_reps is None:
        num_reps = 1
    num_nvs = len(pol_coords_list)

    with qua.program() as seq:
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        def one_rep(rep_ind=None):
            with qua.if_(rep_ind == 0):
                seq_utils.macro_ionize(ion_coords_list)
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

    seq_ret_vals = []
    return seq, seq_ret_vals


def get_conditional_sequence(
    pol_coords_list,
    ion_coords_list,
    num_reps,
    condition_fn=None,
    verify_charge_states=False,
    pol_duration_override=None,
    pol_amp_override=None,
    readout_duration_override=None,
    readout_amp_override=None,
):
    """
    Defines a conditional sequence for charge state readout after polarization/ionization.
    """
    if num_reps is None:
        num_reps = 1

    def one_exp(do_ionize):
        # Check the condition before polarization
        if condition_fn is not None and not condition_fn():
            return

        # Polarization step
        seq_utils.macro_polarize(
            pol_coords_list,
            pol_duration_override,
            pol_amp_override,
            targeted_polarization=verify_charge_states,
            verify_charge_states=verify_charge_states,
            spin_pol=False,
        )

        # Optional ionization step
        if do_ionize:
            seq_utils.macro_ionize(ion_coords_list)

        # Charge state readout
        seq_utils.macro_charge_state_readout(
            readout_duration_override, readout_amp_override
        )

        # Wait for the next trigger
        seq_utils.macro_wait_for_trigger()

    def one_rep(rep_ind=None):
        # Perform two experiments: one with ionization and one without
        for do_ionize in [True, False]:
            one_exp(do_ionize)

    # Handle repetitions and pause sequence at the end
    seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
    seq_utils.macro_pause()


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [110, 109.51847988358679],
                [112, 110.70156405156148],
                [112, 110.70156405156148],
                [112, 110.70156405156148],
            ],
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
