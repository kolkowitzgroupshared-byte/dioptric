# -*- coding: utf-8 -*-
"""
QUA sequence for adaptive NV charge initialization followed by readout-only reps.

The Python experiment controls which NVs are targeted by writing the Boolean
input stream ``_cache_target_list`` between repetitions.

Rep convention
--------------
    rep 0:
        Ionize all NVs, then charge-state readout.

    rep >= 1:
        Run targeted charge polarization on NVs marked True in
        ``_cache_target_list``, then charge-state readout.

For the immediate verification and delayed final readout, the Python callback
writes an all-False target list. Therefore these repetitions perform no charge
polarization and only execute the charge-state readout.

Created July 2026.
"""

from __future__ import annotations

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
    """Build the adaptive charge-memory sequence."""

    if num_reps is None:
        num_reps = 1

    num_nvs = len(pol_coords_list)

    with qua.program() as seq:
        seq_utils.init(num_nvs)
        seq_utils.macro_run_aods()

        def one_rep(rep_ind=None):
            # The first readout starts from an intentionally ionized population.
            with qua.if_(rep_ind == 0):
                seq_utils.macro_ionize(ion_coords_list)

            # All later repetitions use the streamed target mask. During the
            # verification/final readout reps, Python sends all False, so this
            # macro applies no targeted polarization pulses.
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

        seq_utils.handle_reps(
            one_rep,
            num_reps,
            wait_for_trigger=False,
        )
        seq_utils.macro_pause()

    return seq, []


if __name__ == "__main__":
    # Optional local simulation. Replace the example coordinate and pulse lists
    # with values appropriate for your configuration before running directly.
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qmm = QuantumMachinesManager(**config["DeviceIDs"]["QM_opx_args"])
    opx = qmm.open_qm(opx_config)

    try:
        example_coords = [[110.0, 109.5], [112.0, 110.7]]
        example_durations = [1000, 1000]
        example_amps = [1.0, 1.0]

        seq, _ = get_seq(
            ion_coords_list=example_coords,
            pol_coords_list=example_coords,
            pol_duration_list=example_durations,
            pol_amp_list=example_amps,
            num_reps=4,
        )

        simulation = opx.simulate(
            seq,
            SimulationConfig(duration=int(150e3 / 4)),
        )
        simulation.get_simulated_samples().con1.plot()
    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)
