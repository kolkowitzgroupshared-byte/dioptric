import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import tool_belt as tb
from utils import widefield


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    min_tau,
    max_tau,
    uwave_ind_list,
    phase_deg,
):

    phase_deg = int(phase_deg) % 360

    pulse_gen = tb.get_server_pulse_gen()

    seq_file = "rabi_iq_phase.py"

    taus = np.linspace(
        min_tau,
        max_tau,
        num_steps,
    )


    def run_fn(shuffled_step_inds):

        shuffled_taus = [
            taus[ind]
            for ind in shuffled_step_inds
        ]

        seq_args = [
            widefield.get_base_scc_seq_args(
                nv_list,
                uwave_ind_list,
            ),
            shuffled_taus,
            phase_deg,
        ]

        pulse_gen.stream_load(
            seq_file,
            tb.encode_seq_args(seq_args),
            num_reps,
        )


    raw_data = base_routine.main(
        nv_list,
        num_steps=num_steps,
        num_reps=num_reps,
        num_runs=num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )


    timestamp = dm.get_time_stamp()

    raw_data |= {
        "timestamp": timestamp,
        "taus": taus,
        "tau-units": "ns",
        "min_tau": min_tau,
        "max_tau": max_tau,
        "phase_deg": phase_deg,
        "uwave_ind_list": list(uwave_ind_list),
        "experiment": "IQ phase Rabi",
    }


    repr_nv_sig = widefield.get_repr_nv_sig(
        nv_list
    )

    file_path = dm.get_file_path(
        __file__,
        timestamp,
        repr_nv_sig.name,
    )

    dm.save_raw_data(
        raw_data,
        file_path,
    )

    tb.reset_cfm()

    print()
    print(
        f"IQ Rabi phase {phase_deg} deg finished"
    )
    print(
        "Saved:",
        file_path,
    )

    return raw_data