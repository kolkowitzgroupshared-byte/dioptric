# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization, no spin manipulation

Created on October 13th, 2023

@author: mccambria
@author: schand
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def macro(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    ion_coords_list,
    num_reps,
    ion_duration_list=None,
    ion_amp_list=None,
    ion_do_target_list=None,
    verify_charge_states=False,
    pol_duration_override=None,
    pol_amp_override=None,
    ion_duration_override=None,
    ion_amp_override=None,
    readout_duration_override=None,
    readout_amp_override=None,
    aod_accees_time_override=None,
    repeated_readout= False,
):
    if num_reps is None:
        num_reps = 1

    def _rev(x):
        if x is None:
            return None
        return x[::-1]

    def macro_polarize_sub(reverse=False):
        if reverse:
            coords = pol_coords_list[::-1]
            durations = _rev(pol_duration_list)
            amps = _rev(pol_amp_list)
        else:
            coords = pol_coords_list
            durations = pol_duration_list
            amps = pol_amp_list

        seq_utils.macro_polarize(
            coords,
            duration_list=durations,
            amp_list=amps,
            duration_override=pol_duration_override,
            amp_override=pol_amp_override,
            targeted_polarization=verify_charge_states,
            verify_charge_states=verify_charge_states,
            spin_pol=False,
            aod_accees_time_override=aod_accees_time_override,
        )

    def macro_ionize_sub(reverse=False):
        if reverse:
            coords = ion_coords_list[::-1]
            durations = _rev(ion_duration_list)
            amps = _rev(ion_amp_list)
            target_list = _rev(ion_do_target_list)
        else:
            coords = ion_coords_list
            durations = ion_duration_list
            amps = ion_amp_list
            target_list = ion_do_target_list

        seq_utils.macro_ionize(
            coords,
            ion_duration_list=durations,
            ion_amp_list=amps,
            ion_duration_override=ion_duration_override,
            ion_amp_override=ion_amp_override,
            aod_accees_time_override=aod_accees_time_override,
            do_target_list=target_list,
        )

    def one_exp(do_ionize):
        random_order = qua.declare(int)
        qua.assign(random_order, qua.Random().rand_int(2))

        # Important:
        # If verify_charge_states=True, reversing order can mismatch the
        # target-list input stream unless the target list is also reversed
        # on the host side. For normal histogram mode, verify_charge_states
        # is usually False, so reversing is OK.
        if verify_charge_states:
            macro_polarize_sub(reverse=False)

            if do_ionize:
                macro_ionize_sub(reverse=False)
                # Wait after ionization before charge readout
                post_ion_wait_cc = seq_utils.convert_ns_to_cc(200_000)  # 200 us
                qua.align()
                qua.wait(post_ion_wait_cc, "do_camera_trigger")

        else:
            with qua.if_(random_order == 1):
                macro_polarize_sub(reverse=False)

                if do_ionize:
                    macro_ionize_sub(reverse=False)
                    # Wait after ionization before charge readout
                    post_ion_wait_cc = seq_utils.convert_ns_to_cc(200_000)  # 200 us
                    qua.align()
                    qua.wait(post_ion_wait_cc, "do_camera_trigger")

            with qua.else_():
                macro_polarize_sub(reverse=True)

                if do_ionize:
                    macro_ionize_sub(reverse=True)
                    # Wait after ionization before charge readout
                    post_ion_wait_cc = seq_utils.convert_ns_to_cc(200_000)  # 200 us
                    qua.align()
                    qua.wait(post_ion_wait_cc, "do_camera_trigger")

        seq_utils.macro_charge_state_readout(
            readout_duration_override,
            readout_amp_override,
        )
        seq_utils.macro_wait_for_trigger()
        
        # repeated-readout test:
        # Do a second readout immediately after the first one, without any
        # re-polarization or re-ionization. Comparing readout 1 and readout 2
        # tells us how much the first readout perturbs the charge state.
        if repeated_readout:
            seq_utils.macro_charge_state_readout(
                readout_duration_override,
                readout_amp_override,
            )
            seq_utils.macro_wait_for_trigger()
        
    def one_rep(rep_ind=None):
        for do_ionize in [True, False]:
            one_exp(do_ionize)


    seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
    seq_utils.macro_pause()