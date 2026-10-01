# -*- coding: utf-8 -*-
"""
Base spin sequence for widefield experiments with many spatially resolved NV centers.
Accompanies base routine

Created on December 11th, 2023

@author: mccambria
@author: sbchand
"""

from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils


def macro(
    base_scc_seq_args,
    uwave_macro,
    step_val=None,
    num_reps=1,
    scc_duration_override=None,
    scc_amp_override=None,
    spin_pol_duration_override=None,
    spin_pol_amp_override=None,
    readout_duration_override=None,
    readout_amp_override=None,
    aod_accees_time_override = None,
    reference=True,
):
    """Base spin sequence as a QUA macro for widefield experiments with many
    spatially resolved NV centers. Accompanies base routine

    Parameters
    ----------
    pol_coords_list : list(float)
        List of polarization coordinates to target -
        returned by widefield.get_base_scc_seq_args(nv_list)
    ion_coords_list : list(float)
        List of ionization coordinates to target -
        returned by widefield.get_base_scc_seq_args(nv_list)
    num_reps : int
        Number of repetitions to do
    uwave_macro : function or list(function)
        QUA macro describing the microwave pulse sequence. If a list, then
        run multiple experiments per rep - the first macro will be run for
        the first experiment, the second macro for the second experiment, etc.
    pol_duration_ns : int, optional
        Duration of polarization pulse in ns, by default pulls from config
    ion_duration_ns : int, optional
        Duration of ionization pulse in ns, by default pulls from config
    readout_duration_ns : int, optional
        Duration of readout pulse in ns, by default pulls from config
    setup_macro : function, optional
        QUA macro describing any setup sequence we may want to run. Called
        at the very beginning even before turning on AODs. By default None
    reference : bool, optional
        Whether to include a reference experiment in which no microwaves
        are applied, by default True

    Returns
    -------
    _type_
        _description_
    """

    ### Non-QUA stuff
    (
        pol_coords_list,
        pol_duration_list,
        pol_amp_list,
        scc_coords_list,
        scc_duration_list,
        scc_amp_list,
        spin_flip_do_target_list,
        uwave_ind_list,
    ) = base_scc_seq_args

    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]

    if num_reps is None:
        num_reps = 1

    # Construct the list of experiments to run
    if not isinstance(uwave_macro, list):
        uwave_macro = [uwave_macro]
    if reference:

        def ref_exp(uwave_ind_list, step_val):
            pass

        uwave_macro.append(ref_exp)

    num_exps_per_rep = len(uwave_macro)
    num_nvs = len(pol_coords_list)

    def macro_polarize_sub():
        seq_utils.macro_polarize(
            pol_coords_list,
            duration_list=pol_duration_list,
            amp_list=pol_amp_list,
            spin_pol_duration_override=spin_pol_duration_override,
            spin_pol_amp_override= spin_pol_amp_override,
            aod_accees_time_override = aod_accees_time_override
        )
        
    def macro_polarize_sub_reversed():
        seq_utils.macro_polarize(
            pol_coords_list[::-1],
            duration_list=pol_duration_list[::-1],
            amp_list=pol_amp_list[::-1],
            spin_pol_duration_override=spin_pol_duration_override,
            spin_pol_amp_override=spin_pol_amp_override,
            aod_accees_time_override = aod_accees_time_override,
        )

    def macro_scc_sub(do_target_list=None):
        seq_utils.macro_scc(
            scc_coords_list,
            scc_duration_list,
            scc_amp_list,
            scc_duration_override,
            scc_amp_override,
            aod_accees_time_override,
            do_target_list,
        )

    # SBC Reverse the scc order on NVs
    def macro_scc_sub_reversed(do_target_list=None):
        seq_utils.macro_scc(
            scc_coords_list[::-1],
            scc_duration_list[::-1],
            scc_amp_list[::-1],
            scc_duration_override,
            scc_amp_override,
            aod_accees_time_override,
            (do_target_list[::-1] if do_target_list is not None else None),
        )

    ### QUA stuff
    def one_exp(rep_ind, exp_ind):
        # exp_ind = num_exps_per_rep - 1  # MCC
        # macro_polarize_sub()
        
        #SBC: qua variable for randomize Pol and SCC order
        random_order = qua.declare(int)
        qua.assign(random_order, qua.Random().rand_int(2))

        with qua.if_(random_order == 1):
            macro_polarize_sub()
        with qua.else_():
             macro_polarize_sub_reversed()
        
        qua.align()
        
        skip_spin_flip = uwave_macro[exp_ind](uwave_ind_list, step_val)

        # Check if this is the automatically included reference experiment
        ref_exp = reference and exp_ind == num_exps_per_rep - 1
        print(f"exp_ind: {exp_ind}, ref_exp: {ref_exp}")
        # Signal experiment
        if not ref_exp:
            if spin_flip_do_target_list is None or True not in spin_flip_do_target_list:
                # macro_scc_sub()  # do scc alwayd in the order of NVs
                # SBC randomize the order of the scc by alterntively reversing the order
                with qua.if_(random_order == 1):
                    macro_scc_sub()
                with qua.else_():
                    macro_scc_sub_reversed()
            else:
                spin_flip_do_not_target_list = [
                    not val for val in spin_flip_do_target_list
                ]
                # Randomized SCC order between the two groups
                with qua.if_(random_order == 1):
                    macro_scc_sub(spin_flip_do_not_target_list)
                    if not skip_spin_flip:
                        seq_utils.macro_pi_pulse(uwave_ind_list)
                    macro_scc_sub(spin_flip_do_target_list)
                with qua.else_():
                    macro_scc_sub(spin_flip_do_target_list)
                    if not skip_spin_flip:
                        seq_utils.macro_pi_pulse(uwave_ind_list)
                    macro_scc_sub(spin_flip_do_not_target_list)

        # Reference experiment
        else:
            # "Dual-rail" referencing: measure ms=0 for even reps, and ms=+/-1
            # for odd by applying an extra pi pulse just before SCC
            with qua.if_(qua.Cast.unsafe_cast_bool(rep_ind)):
                # seq_utils.macro_pi_pulse(uwave_ind_list, phase=0)
                seq_utils.macro_pi_pulse(uwave_ind_list)
            # macro_scc_sub()
            #  SBC randomize the order of the scc by alterntively reversing the order
            with qua.if_(random_order == 1):
                macro_scc_sub()
            with qua.else_():
                macro_scc_sub_reversed()

        seq_utils.macro_charge_state_readout(
            readout_duration_override, readout_amp_override
        )
        seq_utils.macro_wait_for_trigger()

    def one_rep(rep_ind=0):
        print(num_exps_per_rep)
        for exp_ind in range(num_exps_per_rep):
            one_exp(rep_ind, exp_ind)

    seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
    seq_utils.macro_pause()
