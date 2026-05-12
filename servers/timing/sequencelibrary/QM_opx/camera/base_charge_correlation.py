# -*- coding: utf-8 -*-
"""
Base macro for equal-time charge-state correlation measurement.
"""

from servers.timing.sequencelibrary.QM_opx import seq_utils


def macro(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    do_drive,
    targeted_drive,
    num_reps,
    pol_duration_override=None,
    pol_amp_override=None,
    readout_duration_override=None,
    readout_amp_override=None,
):
    if num_reps is None:
        num_reps = 1

    def one_rep(rep_ind=None):
        if do_drive:
            seq_utils.macro_polarize(
                pol_coords_list,
                pol_duration_list,
                pol_amp_list,
                pol_duration_override,
                pol_amp_override,
                targeted_polarization=targeted_drive,
                verify_charge_states=False,
                spin_pol=False,
            )

        seq_utils.macro_charge_state_readout(
            readout_duration_override,
            readout_amp_override,
        )

        seq_utils.macro_wait_for_trigger()

    seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
    seq_utils.macro_pause()