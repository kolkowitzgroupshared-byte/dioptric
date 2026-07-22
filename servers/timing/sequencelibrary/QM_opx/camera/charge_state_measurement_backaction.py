# -*- coding: utf-8 -*-
"""
QUA sequence for measurement-induced NV charge-state transitions.

The sequence supports:

    1. Adaptive initialization of the NVs into NV-.
    2. An immediate verification readout.
    3. Repeated charge-state readouts without reinitialization.

The Python experiment controls which NVs receive charge-polarization pulses by
writing a Boolean list to the input stream ``_cache_target_list`` before each
repetition.

Repetition convention
---------------------
    rep 0:
        Ionize all NVs and perform charge-state readout.

    initialization reps:
        Python sends an adaptive target mask.
        Only NVs marked True receive charge polarization.

    immediate verification:
        Python sends an all-False target mask.
        No charge polarization is applied; only readout is performed.

    delayed measurements:
        Python waits for the requested dark interval and then sends an
        all-False target mask.
        No reinitialization is performed; only readout is performed.

Important
---------
The long dark interval, for example 300 s, is controlled by the Python
experiment callback and is not implemented as a long QUA wait.

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
    """
    Build the repeated charge-state measurement sequence.

    Parameters
    ----------
    ion_coords_list
        AOD coordinates used to ionize all selected NVs during rep 0.

    pol_coords_list
        AOD coordinates used for targeted NV- charge polarization.

    pol_duration_list
        Charge-polarization duration for each NV.

    pol_amp_list
        Charge-polarization amplitude for each NV.

    num_reps
        Total number of repetitions, including initialization,
        immediate verification, and delayed readouts.
    """

    if num_reps is None:
        num_reps = 1

    num_reps = int(num_reps)
    num_nvs = len(pol_coords_list)

    if num_reps < 1:
        raise ValueError(
            "num_reps must be at least 1."
        )

    if len(ion_coords_list) != num_nvs:
        raise ValueError(
            "ion_coords_list and pol_coords_list must have the same length."
        )

    if len(pol_duration_list) != num_nvs:
        raise ValueError(
            "pol_duration_list and pol_coords_list must have the same length."
        )

    if len(pol_amp_list) != num_nvs:
        raise ValueError(
            "pol_amp_list and pol_coords_list must have the same length."
        )

    with qua.program() as seq:
        # Initializes QUA variables, input streams, and camera-related state.
        seq_utils.init(num_nvs)

        # Start the AOD carrier tones used by the coordinate macros.
        seq_utils.macro_run_aods()

        def one_rep(rep_ind=None):
            """
            Execute one experiment repetition.

            rep 0 intentionally prepares an ionized starting population.
            Every later repetition uses the streamed Boolean target mask.
            """

            # ----------------------------------------------------------
            # First repetition: intentionally ionize all NVs
            # ----------------------------------------------------------
            with qua.if_(rep_ind == 0):
                seq_utils.macro_ionize(
                    ion_coords_list
                )

            # ----------------------------------------------------------
            # All subsequent repetitions
            # ----------------------------------------------------------
            with qua.else_():
                # During adaptive initialization, Python sends True for NVs
                # that still require charge polarization.
                #
                # During immediate verification and all delayed readouts,
                # Python sends an all-False list. Therefore no NV receives
                # a polarization pulse and this becomes a readout-only rep.
                seq_utils.macro_polarize(
                    pol_coords_list,
                    pol_duration_list,
                    pol_amp_list,
                    spin_pol=False,
                    targeted_polarization=True,
                    verify_charge_states=False,
                )

            # Charge-state image/readout for every repetition.
            seq_utils.macro_charge_state_readout()

            # Synchronize the sequence with the camera/base routine.
            seq_utils.macro_wait_for_trigger()

        seq_utils.handle_reps(
            one_rep,
            num_reps,
            wait_for_trigger=False,
        )

        # Keep the QUA program available until the host finishes the run.
        seq_utils.macro_pause()

    return seq, []


# =============================================================================
# Optional local simulation
# =============================================================================

if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qmm = QuantumMachinesManager(
        **config["DeviceIDs"]["QM_opx_args"]
    )

    opx = qmm.open_qm(
        opx_config
    )

    try:
        example_coords = [
            [110.0, 109.5],
            [112.0, 110.7],
        ]

        example_durations = [
            1000,
            1000,
        ]

        example_amps = [
            1.0,
            1.0,
        ]

        # Example interpretation:
        #
        # rep 0: ionize + readout
        # rep 1: adaptive initialization + readout
        # rep 2: adaptive initialization + readout
        # rep 3: immediate readout-only measurement
        # rep 4: delayed readout-only measurement
        # rep 5: delayed readout-only measurement
        seq, _ = get_seq(
            ion_coords_list=example_coords,
            pol_coords_list=example_coords,
            pol_duration_list=example_durations,
            pol_amp_list=example_amps,
            num_reps=6,
        )

        simulation = opx.simulate(
            seq,
            SimulationConfig(
                duration=int(200e3 / 4)
            ),
        )

        simulation.get_simulated_samples().con1.plot()

    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)