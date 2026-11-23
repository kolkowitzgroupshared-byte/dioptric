# -*- coding: utf-8 -*-
"""
Widefield XY4-N with a phase-locked test tone: calibrates the AC field per OPX volt at
each NV directly at an in-band frequency, through the same chain RQS uses.

Works like spin_echo_tone.py (tone amplitude swept at two tone phases 90 deg apart; each
NV's signal follows c + b cos(k A)), with the Hahn echo replaced by n_pi pi pulses in
XY4 order, half a tone period apart. The sequence then senses the tone for its whole
length n_pi / (2f), so the field per volt is k_max / (gamma (2/pi) n_pi / (2f)).

Why XY4 at 100 kHz with n_pi = 12: the pi pulses 5 us apart lock onto 100 kHz, and the
60 us length is two 13C precession periods (13C at ~33.3 kHz here), so the 13C field
cancels while the tone keeps adding phase. Static offsets cancel too. Same sensitivity
per nT as the 16.7 kHz Hahn echo (also 60 us), so the same amplitudes and attenuation work.

The printed tone phase offset is relative to this sequence's cos convention, which
differs by 90 deg from the Hahn echo's.

Control panel function (after: from majorroutines.widefield import xy_tone):

def do_xy_tone(nv_list):
    amps_v = np.linspace(0, 0.5, 21).tolist()
    xy_tone.main(nv_list, num_reps=10, num_runs=30, amps_v=amps_v,
                 freq_hz=100e3, n_pi=12, attenuation_db=30)

Created on November 15th, 2026
"""

import numpy as np

from majorroutines.widefield.spin_echo_tone import analyze


def main(nv_list, num_reps, num_runs, amps_v, freq_hz=100e3, n_pi=12, phases_deg=(90.0, 0.0),
         attenuation_db=None, uwave_ind_list=(1,)):
    """amps_v: tone amplitudes (V into 50 ohm at the OPX output), together within 0.5 V.
    n_pi: pi pulses in XY4 order, half a tone period apart.
    attenuation_db: what is in the chain during this run, recorded with the data."""
    from majorroutines.widefield import base_routine
    from utils import data_manager as dm
    from utils import kplotlib as kpl
    from utils import tool_belt as tb
    from utils import widefield

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "xy_tone.py"
    uwave_ind_list = [int(ind) for ind in uwave_ind_list]
    amps_v = [float(amp) for amp in amps_v]
    phases_deg = [float(phase) for phase in phases_deg]
    n_pi = int(n_pi)
    num_steps = len(phases_deg) * len(amps_v)

    def run_fn(shuffled_step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            [int(ind) for ind in shuffled_step_inds],
            float(freq_hz),
            phases_deg,
            amps_v,
            n_pi,
        ]
        pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args), num_reps)

    # load_iq=True puts the signal generator in IQ mode, so the 90 deg (Y) pulses are real
    # Y rotations, as in xy.py
    raw_data = base_routine.main(
        nv_list, num_steps, num_reps, num_runs, run_fn, uwave_ind_list=uwave_ind_list,
        load_iq=True,
    )

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "amps_v": amps_v,
        "amps_v-units": "V into 50 ohm at the OPX output",
        "phases_deg": phases_deg,
        "freq_hz": float(freq_hz),
        "n_pi": n_pi,
        "pi_phases_deg": [(0, 90, 0, 90)[k % 4] for k in range(n_pi)],
        "sensing_time_s": n_pi / (2 * float(freq_hz)),
        "attenuation_db": attenuation_db,
        "step_order": "step = phase_index * len(amps_v) + amplitude_index",
    }

    fig = None
    try:
        results = analyze(raw_data)
        fig = results.pop("figure")
        raw_data |= results
    except Exception as exc:
        print(f"Analysis failed ({exc}); the raw data is still saved.")

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_sig.name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)

    tb.reset_cfm()
    kpl.show()
    return raw_data


if __name__ == "__main__":
    from majorroutines.widefield.spin_echo_tone import analyze, save_reanalysis
    from utils import data_manager as dm
    from utils import kplotlib as kpl

    FILE_STEM = "2026_11_18-16_10_31-johnson-nv0_2026_10_02"  # XY2, stopped by an OPX timeout
    RUNS_DONE = 6  # runs 0-5 finished before the timeout

    kpl.init_kplotlib()
    data = dm.get_raw_data(file_stem=FILE_STEM, load_npz=True)
    data["counts"] = np.asarray(data["counts"])[:, :, :RUNS_DONE]  # drop the unfilled runs
    results = analyze(data, details=True)
    save_reanalysis(results, FILE_STEM, __file__)
    kpl.show(block=True)