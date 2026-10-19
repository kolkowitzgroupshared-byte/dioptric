from qm import qua

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(
    base_scc_seq_args,
    step_vals,
    phase_deg,
    num_reps=1,
):
    step_vals = [
        seq_utils.convert_ns_to_cc(el)
        for el in step_vals
    ]

    phase_deg = int(phase_deg) % 360

    if phase_deg not in (0, 90, 180, 270):
        raise ValueError(
            "phase_deg must be 0, 90, 180, or 270"
        )

    with qua.program() as seq:

        seq_utils.init()
        seq_utils.macro_run_aods()

        step_val = qua.declare(int)

        def uwave_macro_sig(
            uwave_ind_list,
            step_val,
        ):
            qua.align()

            seq_utils.macro_pi_pulse(
                uwave_ind_list,
                duration_cc=step_val,
                phase=phase_deg,
            )

            qua.align()

        with qua.for_each_(
            step_val,
            step_vals,
        ):

            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                step_val,
                num_reps,
            )

    return seq, []