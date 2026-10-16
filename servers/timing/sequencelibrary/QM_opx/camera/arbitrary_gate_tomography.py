"""Cornell-Berkeley Experiment 0, OPX camera sequence.

Follows camera/bootstrapped_pulse_error_tomography.py + base_scc_sequence.
One measurement frame contains 3 * n_gates signals (Z,X,Y per gate) and
one dual-rail reference. The reference alternates |0> (even rep) and
|1> (odd rep), as in the existing base_scc_sequence.
"""
from qm import qua
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, compiled_gates, num_reps=1):
    if num_reps < 2:
        raise ValueError("Dual-rail SCC referencing needs >=2 reps (4+ recommended)")
    mw_chains = base_scc_seq_args[-1]
    if len(mw_chains) != 1:
        raise ValueError("Experiment 0 requires one addressed MW chain at a time")
    for gate in compiled_gates:
        for p in gate["pulses"]:
            if p["phase_deg"] not in (0, 90, 180, 270) or p["duration_cc"] < 4:
                raise ValueError(f"Invalid precompiled pulse: {p}")

    def make_macro(pulses, basis):
        # Compile-time closure; no per-rep host updates / no QUA dynamic arrays.
        def microwave_macro(uwave_ind_list, step_val):
            qua.align()
            for p in pulses:
                seq_utils.macro_pi_pulse(
                    uwave_ind_list,
                    duration_cc=int(p["duration_cc"]),
                    phase=int(p["phase_deg"]),
                )
            # In the standard Pauli frame:
            # <X> readout = R_y(-pi/2) then Z
            # <Y> readout = R_x(+pi/2) then Z
            if basis == "X":
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=270)
            elif basis == "Y":
                seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=0)
            elif basis != "Z":
                raise ValueError(f"Unknown readout basis {basis}")
            qua.align()
        return microwave_macro

    macro_list = []
    # This ordering must match the experiment and analysis metadata.
    for gate in compiled_gates:
        for basis in ("Z", "X", "Y"):
            macro_list.append(make_macro(gate["pulses"], basis))

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        base_scc_sequence.macro(
            base_scc_seq_args, macro_list,
            num_reps=num_reps,
            reference=True,  # last experiment is the alternating dual-rail ref
        )
    return seq, []
