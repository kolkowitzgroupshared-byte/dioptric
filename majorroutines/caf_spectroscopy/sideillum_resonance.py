# -*- coding: utf-8 -*-
"""
Confocal ESR experiment using base routine.
Sweeps microwave frequency, reads signal and reference counts via APD tagger.

Adapted to CaF2 code.

Created on Mar 16, 2026
@author: j-chen1
"""

import numpy as np

from majorroutines.caf_spectroscopy.sideillum_base_routine import main as base_routine
from utils import data_manager as dm
from utils import tool_belt as tb


def main(
    coords,
    freqs,
    num_reps,
    num_runs,
    apd_ch,
    apd_time,
    use_reference=True,
    norm_style="contrast",
):
    """
    Parameters:
        seq_file: seq program path
        scan_coords: [x, y, z] center
        freqs: array of frequencies in GHz
        num_reps: repetitions per point
        num_runs: number of full sweeps
        apd_ch: APD channel
        apd_time: APD collection time in seconds
        run_nir_fn: function(bool) to toggle NIR (optional)
        use_reference: whether to use signal/reference gates
    """
    pulse_streamer = tb.get_server_pulse_streamer()
    tagger = tb.get_server_time_tagger()

    seq_file = "resonance_caf.py"

    def apd_read_fn(tagger, apd_ch, apd_time):
        if use_reference:
            counts = tb.read_apd_2gates(tagger, apd_ch, apd_time)
        else:
            counts = tb.read_apd_counts(tagger, apd_ch, apd_time)
        return counts

    # Call the base routine
    raw_data = base_routine(
        scan_coords=coords,
        num_steps=len(freqs),
        num_reps=num_reps,
        num_runs=num_runs,
        # seq_args_fn=seq_args_fn,
        apd_read_fn=apd_read_fn,  # TODO: ??
        tagger=tagger,
        apd_ch=apd_ch,
        apd_time=apd_time,
    )

    # Optional normalization
    # if use_reference:
    #     raw_counts = np.array(raw_data["counts"])
    #     norm, ste = tb.process_counts_array(
    #         raw_counts, gate_mode="2gate", norm_style=norm_style
    #     )
    #     raw_data["norm"] = norm.tolist()
    #     raw_data["norm_ste"] = ste.tolist()
    #     raw_data["norm_style"] = norm_style

    raw_data["freqs"] = freqs.tolist()
    raw_data["use_reference"] = use_reference
    dm.save_raw_data(raw_data, dm.get_file_path(__file__, raw_data["timestamp"]))
    return raw_data


if __name__ == "__main__":
    main()
