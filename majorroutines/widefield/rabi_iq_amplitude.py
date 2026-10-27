"""Fixed-duration, amplitude-swept IQ Rabi with widefield SCC.

Amplitude is the commanded IQ vector amplitude at the OPX DAC (0..0.5 V);
the SG394 microwave power and frequency remain configured as before.

Example:
    from majorroutines.widefield import rabi_iq_amplitude
    raw = rabi_iq_amplitude.main(
        nv_list, num_steps=41, num_reps=200, num_runs=10,
        pulse_duration_ns=384, uwave_ind_list=(1,),
        min_amp_v=0.0, max_amp_v=0.5, phase_deg=0,
    )
"""
import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import tool_belt as tb
from utils import widefield

FULL_SCALE_V = 0.5
SEQ_FILE = "rabi_iq_amplitude.py"


def make_sweep(num_steps, min_amp_v=0., max_amp_v=0.5):
    if isinstance(num_steps, bool) or int(num_steps) != num_steps or num_steps < 2:
        raise ValueError("num_steps must be integer >= 2")
    if not (np.isfinite(min_amp_v) and np.isfinite(max_amp_v)
            and 0 <= min_amp_v < max_amp_v <= FULL_SCALE_V):
        raise ValueError("Require 0 <= min_amp_v < max_amp_v <= 0.5 V")
    return np.linspace(min_amp_v, max_amp_v, int(num_steps), dtype=float)


def main(nv_list, num_steps=41, num_reps=200, num_runs=10,
         pulse_duration_ns=384, uwave_ind_list=(1,),
         min_amp_v=0.0, max_amp_v=0.5, phase_deg=0):
    if not nv_list:
        raise ValueError("nv_list is empty")
    if len(uwave_ind_list) != 1:
        raise ValueError("Select exactly one microwave chain (e.g., (1,))")
    if num_reps < 2 or num_reps % 2:
        raise ValueError("Use even num_reps >= 2 for alternating SCC references")
    if num_runs < 1 or int(num_runs) != num_runs:
        raise ValueError("num_runs must be integer >=1")
    if pulse_duration_ns < 16 or int(pulse_duration_ns) != pulse_duration_ns or pulse_duration_ns % 4:
        raise ValueError("pulse_duration_ns must be a multiple of 4 ns >=16")
    if int(phase_deg) != phase_deg or not 0 <= phase_deg < 360:
        raise ValueError("phase_deg must be an integer from 0 through 359")
    amplitudes = make_sweep(num_steps, min_amp_v, max_amp_v)

    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        # base_routine controls step randomization; transmit in its exact order.
        shuffled_amps = [float(amplitudes[i]) for i in shuffled_step_inds]
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, list(uwave_ind_list)),
            shuffled_amps,
            int(pulse_duration_ns),
            int(phase_deg),
        ]
        pulse_gen.stream_load(
            SEQ_FILE, tb.encode_seq_args(seq_args), num_reps
        )

    try:
        raw = base_routine.main(
            nv_list,
            num_steps=int(num_steps),
            num_reps=int(num_reps),
            num_runs=int(num_runs),
            run_fn=run_fn,
            uwave_ind_list=list(uwave_ind_list),
            load_iq=True,
        )

        timestamp = dm.get_time_stamp()
        raw.update({
            "timestamp": timestamp,
            "experiment": "Fixed-duration IQ amplitude Rabi",
            "iq_amplitudes_v": amplitudes.tolist(),
            "iq_amplitude_units": "V (OPX I/Q vector amplitude)",
            "iq_waveform_full_scale_v": FULL_SCALE_V,
            "iq_qua_scales": (amplitudes / FULL_SCALE_V).tolist(),
            "min_amp_v": float(min_amp_v),
            "max_amp_v": float(max_amp_v),
            "pulse_duration_ns": int(pulse_duration_ns),
            "phase_deg": int(phase_deg),
            "uwave_ind_list": list(uwave_ind_list),
            "reference_protocol": "dual_rail_even_ms0_odd_ms1",
            "num_exps_per_step": 2,
            "zero_amplitude_skips_microwave_ttl": True,
        })
        repr_nv = widefield.get_repr_nv_sig(nv_list)
        file_path = dm.get_file_path(__file__, timestamp, repr_nv.name)
        # Preserve the original raw 'counts' ndarray: dm may mutate data.
        dm.save_raw_data(raw.copy(), file_path)
        print(f"IQ amplitude Rabi saved: {file_path}")
        return raw
    finally:
        tb.reset_cfm()
