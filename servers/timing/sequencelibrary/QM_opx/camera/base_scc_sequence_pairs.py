# -*- coding: utf-8 -*-
"""
Base spin sequence for widefield experiments with many spatially resolved NV centers.

This version preserves the original working base_scc_sequence behavior, but adds
optional row-pair multiplexed charge initialization and SCC.

Optional final entry in base_scc_seq_args:

    {
        "pairwise_init": True,
        "pairwise_scc": True,
        "y_tol_MHz": 0.05,
        "spin_pol": True,
    }

If no options dict is provided, this behaves like the original serial sequence.

@author: mccambria
@author: sbchand
"""

from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils


def _unpack_base_scc_seq_args(base_scc_seq_args):
    """
    Backward-compatible unpacker.

    Old format:
        [
            pol_coords_list,
            pol_duration_list,
            pol_amp_list,
            scc_coords_list,
            scc_duration_list,
            scc_amp_list,
            spin_flip_do_target_list,
            uwave_ind_list,
        ]

    New optional format:
        same as above, plus final opts dict.
    """

    args = list(base_scc_seq_args)

    opts = {}
    if len(args) >= 9 and isinstance(args[-1], dict):
        opts = args.pop()

    # Optional list-style fallback if dict encoding gives trouble.
    elif len(args) >= 9 and isinstance(args[-1], list):
        extra = args.pop()
        opts = {
            "pairwise_init": bool(extra[0]),
            "pairwise_scc": bool(extra[1]),
            "y_tol_MHz": float(extra[2]),
            "spin_pol": bool(extra[3]),
        }

    if len(args) != 8:
        raise RuntimeError(
            "base_scc_seq_args must have 8 entries, or 9 entries with final options."
        )

    (
        pol_coords_list,
        pol_duration_list,
        pol_amp_list,
        scc_coords_list,
        scc_duration_list,
        scc_amp_list,
        spin_flip_do_target_list,
        uwave_ind_list,
    ) = args

    return (
        pol_coords_list,
        pol_duration_list,
        pol_amp_list,
        scc_coords_list,
        scc_duration_list,
        scc_amp_list,
        spin_flip_do_target_list,
        uwave_ind_list,
        opts,
    )


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
    aod_accees_time_override=None,
    reference=True,
):
    """
    Base spin sequence as a QUA macro.

    This version keeps the original sequence order:

        charge polarization
        spin polarization
        microwave / pi pulse
        SCC
        charge readout

    Pairwise mode only changes how charge polarization and SCC are delivered.
    """

    # ------------------------------------------------------------------
    # Non-QUA unpacking
    # ------------------------------------------------------------------
    (
        pol_coords_list,
        pol_duration_list,
        pol_amp_list,
        scc_coords_list,
        scc_duration_list,
        scc_amp_list,
        spin_flip_do_target_list,
        uwave_ind_list,
        opts,
    ) = _unpack_base_scc_seq_args(base_scc_seq_args)

    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]

    if num_reps is None:
        num_reps = 1

    pairwise_init = bool(opts.get("pairwise_init", False))
    pairwise_scc = bool(opts.get("pairwise_scc", False))
    y_tol_MHz = float(opts.get("y_tol_MHz", 0.05))

    # Keep spin-pol on by default, same as original macro_polarize behavior.
    spin_pol = bool(opts.get("spin_pol", True))

    # ------------------------------------------------------------------
    # Experiment list
    # ------------------------------------------------------------------
    if not isinstance(uwave_macro, list):
        uwave_macro = [uwave_macro]

    if reference:

        def ref_exp(uwave_ind_list, step_val):
            pass

        uwave_macro.append(ref_exp)

    num_exps_per_rep = len(uwave_macro)

    # ------------------------------------------------------------------
    # Polarization helpers
    # ------------------------------------------------------------------
    def macro_polarize_sub():
        if pairwise_init:
            seq_utils.macro_polarize_row_pairs(
                coords_list=pol_coords_list,
                duration_list=pol_duration_list,
                amp_list=pol_amp_list,
                duration_override=None,
                amp_override=None,  # do not override; use per-NV amps averaged per pair
                targeted_polarization=False,
                verify_charge_states=False,
                spin_pol=spin_pol,
                spin_pol_duration_override=spin_pol_duration_override,
                spin_pol_amp_override=spin_pol_amp_override,
                aod_accees_time_override=aod_accees_time_override,
                y_tol_MHz=y_tol_MHz,
            )
        else:
            seq_utils.macro_polarize(
                pol_coords_list,
                duration_list=pol_duration_list,
                amp_list=pol_amp_list,
                spin_pol=spin_pol,
                spin_pol_duration_override=spin_pol_duration_override,
                spin_pol_amp_override=spin_pol_amp_override,
                aod_accees_time_override=aod_accees_time_override,
            )

    def macro_polarize_sub_reversed():
        if pairwise_init:
            seq_utils.macro_polarize_row_pairs(
                coords_list=pol_coords_list[::-1],
                duration_list=pol_duration_list[::-1],
                amp_list=pol_amp_list[::-1],
                duration_override=None,
                amp_override=None,
                targeted_polarization=False,
                verify_charge_states=False,
                spin_pol=spin_pol,
                spin_pol_duration_override=spin_pol_duration_override,
                spin_pol_amp_override=spin_pol_amp_override,
                aod_accees_time_override=aod_accees_time_override,
                y_tol_MHz=y_tol_MHz,
            )
        else:
            seq_utils.macro_polarize(
                pol_coords_list[::-1],
                duration_list=pol_duration_list[::-1],
                amp_list=pol_amp_list[::-1],
                spin_pol=spin_pol,
                spin_pol_duration_override=spin_pol_duration_override,
                spin_pol_amp_override=spin_pol_amp_override,
                aod_accees_time_override=aod_accees_time_override,
            )

    # ------------------------------------------------------------------
    # SCC helpers
    # ------------------------------------------------------------------
    def macro_scc_sub(do_target_list=None):
        if pairwise_scc:
            seq_utils.macro_scc_row_pairs(
                scc_coords_list=scc_coords_list,
                scc_duration_list=scc_duration_list,
                scc_amp_list=scc_amp_list,
                scc_duration_override=scc_duration_override,
                scc_amp_override=scc_amp_override,
                aod_accees_time_override=aod_accees_time_override,
                do_target_list=do_target_list,
                y_tol_MHz=y_tol_MHz,
            )
        else:
            seq_utils.macro_scc(
                scc_coords_list,
                scc_duration_list,
                scc_amp_list,
                scc_duration_override,
                scc_amp_override,
                aod_accees_time_override,
                do_target_list,
            )

    def macro_scc_sub_reversed(do_target_list=None):
        if do_target_list is not None:
            do_target_list_rev = do_target_list[::-1]
        else:
            do_target_list_rev = None

        if pairwise_scc:
            seq_utils.macro_scc_row_pairs(
                scc_coords_list=scc_coords_list[::-1],
                scc_duration_list=scc_duration_list[::-1],
                scc_amp_list=scc_amp_list[::-1],
                scc_duration_override=scc_duration_override,
                scc_amp_override=scc_amp_override,
                aod_accees_time_override=aod_accees_time_override,
                do_target_list=do_target_list_rev,
                y_tol_MHz=y_tol_MHz,
            )
        else:
            seq_utils.macro_scc(
                scc_coords_list[::-1],
                scc_duration_list[::-1],
                scc_amp_list[::-1],
                scc_duration_override,
                scc_amp_override,
                aod_accees_time_override,
                do_target_list_rev,
            )

    # ------------------------------------------------------------------
    # QUA experiment
    # ------------------------------------------------------------------
    def one_exp(rep_ind, exp_ind):
        # Randomize serial order, same as your original version.
        random_order = qua.declare(int)
        qua.assign(random_order, qua.Random().rand_int(2))

        # --------------------------------------------------------------
        # 1. Charge initialization + spin polarization
        # --------------------------------------------------------------
        with qua.if_(random_order == 1):
            macro_polarize_sub()
        with qua.else_():
            macro_polarize_sub_reversed()

        qua.align()

        # --------------------------------------------------------------
        # 2. Microwave pulse sequence
        # This happens AFTER spin polarization because macro_polarize()
        # already includes macro_spin_polarize() at the end.
        # --------------------------------------------------------------
        skip_spin_flip = uwave_macro[exp_ind](uwave_ind_list, step_val)

        # Check if this is the automatically included reference experiment.
        ref_exp = reference and exp_ind == num_exps_per_rep - 1

        # --------------------------------------------------------------
        # 3. Signal experiment SCC
        # --------------------------------------------------------------
        if not ref_exp:
            if spin_flip_do_target_list is None or True not in spin_flip_do_target_list:
                with qua.if_(random_order == 1):
                    macro_scc_sub()
                with qua.else_():
                    macro_scc_sub_reversed()

            else:
                spin_flip_do_not_target_list = [
                    not val for val in spin_flip_do_target_list
                ]

                # Preserve your original two-group spin-flip SCC logic.
                # Note: if do_target_list is passed, macro_scc_row_pairs()
                # should fall back to serial SCC for safety.
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

        # --------------------------------------------------------------
        # 4. Reference experiment SCC
        # --------------------------------------------------------------
        else:
            # Dual-rail reference:
            # odd/even rep gets extra pi pulse before SCC.
            with qua.if_(qua.Cast.unsafe_cast_bool(rep_ind)):
                seq_utils.macro_pi_pulse(uwave_ind_list)

            with qua.if_(random_order == 1):
                macro_scc_sub()
            with qua.else_():
                macro_scc_sub_reversed()

        # --------------------------------------------------------------
        # 5. Widefield charge readout
        # --------------------------------------------------------------
        seq_utils.macro_charge_state_readout(
            readout_duration_override,
            readout_amp_override,
        )

        seq_utils.macro_wait_for_trigger()

    def one_rep(rep_ind=0):
        for exp_ind in range(num_exps_per_rep):
            one_exp(rep_ind, exp_ind)

    seq_utils.handle_reps(
        one_rep,
        num_reps,
        wait_for_trigger=False,
    )

    seq_utils.macro_pause()