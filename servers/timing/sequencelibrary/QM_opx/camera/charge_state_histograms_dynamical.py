# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, with support for dynamic laser power adjustments.
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(
    pol_coords_list,
    ion_coords_list,
    diff_polarize,
    diff_ionize,
    verify_charge_states,
    pol_laser_power,
    readout_laser_power,
    num_reps,
):
    """
    Generates a QUA sequence for charge state readout after polarization/ionization.
    Supports optional dynamic laser power control.
    """
    if num_reps is None:
        num_reps = 1
    num_nvs = len(pol_coords_list)

    # Set up polarization and ionization logic
    if diff_polarize and not diff_ionize:
        do_polarize_sig = True
        do_polarize_ref = False
        do_ionize_sig = False
        do_ionize_ref = False
    elif not diff_polarize and diff_ionize:
        do_polarize_sig = True
        do_polarize_ref = True
        do_ionize_sig = True
        do_ionize_ref = False
    else:
        do_polarize_sig = do_polarize_ref = do_ionize_sig = do_ionize_ref = True

    with qua.program() as seq:
        # Initialize variables and setup
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        # Define a single experimental loop
        def one_exp(do_polarize_sub, do_ionize_sub):
            # Apply polarization if required
            if do_polarize_sub:
                seq_utils.macro_polarize(
                    pol_coords_list,
                    spin_pol=False,
                    targeted_polarization=verify_charge_states,
                    verify_charge_states=verify_charge_states,
                    polarization_amp=pol_laser_power,  # Use passed power value
                )

            # Apply ionization if required
            if do_ionize_sub:
                seq_utils.macro_ionize(ion_coords_list)

            # Charge state readout with optional readout laser power
            seq_utils.macro_charge_state_readout(amp=readout_laser_power)
            seq_utils.macro_wait_for_trigger()

        # Repetition loop
        def one_rep(rep_ind=None):
            for half_rep_args in [
                [do_polarize_sig, do_ionize_sig],
                [do_polarize_ref, do_ionize_ref],
            ]:
                one_exp(*half_rep_args)

        # Execute the repetitions
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
        # Example coordinates and settings
        seq, seq_ret_vals = get_seq(
            [
                [110, 109.51847988358679],
                [112, 110.70156405156148],
            ],
            [
                [75.42725784791932, 75.65982013416432],
                [75.98725784791932, 74.74382013416432],
            ],
            False,
            True,
            False,
            5,
            pol_laser_power=None,
            readout_laser_power=0.39,
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
