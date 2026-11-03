# -*- coding: utf-8 -*-
"""
Widefield randomized quantum sensing (RQS)

One step is one shot of one acquisition row of the RQS plan: the preparation pi/2,
the row's refocusing pi pulses at the times computed by the RQS toolkit, and the
analysis pi/2. With phase cycling, every row is also played with the analysis pi/2
shifted by 180 deg. All timings come from the files written by
rqs_opx_export_with_pi2 (the <bundle>_opx_timing folder).

Adapted from spin_echo.py on November 3rd, 2026

@author: schand
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

# Rotation axis of each pulse, as a microwave phase in turns
AXIS_TURNS = {"+X": 0.0, "+Y": 0.25, "-X": 0.5, "-Y": 0.75}
# Subtracted from every wait if the simulation shows a fixed extra delay per pulse
WAIT_CORRECTION_CC = 0


def load_rqs_rows(timing_dir, rows):
    """Waits (clock cycles) and pulse axes of the given acquisition rows.

    The waits of rows[j] are waits[offsets[j]:offsets[j + 1]]: one before each pi
    pulse, then one before the analysis pi/2.
    """
    timing_dir = Path(timing_dir)
    acquisition = pd.read_csv(timing_dir / "opx_ready_acquisition.csv")
    gates = pd.read_csv(timing_dir / "opx_full_microwave_sequence.csv")
    axes = gates["nominal_rotation_axis"].astype(str).str.replace("−", "-").str.strip()
    operation = gates["operation"].astype(str)
    prep_axes = set(axes[operation.str.startswith("PREPARE")])
    pi_axes = set(axes[operation == "REFOCUS"])
    if len(prep_axes) != 1 or len(pi_axes) != 1:
        raise ValueError("Expected one preparation axis and one pi-pulse axis for all rows")
    is_analysis = operation.str.startswith("MEASURE")
    analysis_axis = dict(zip(gates["row_id"][is_analysis], axes[is_analysis]))

    waits, offsets, analysis_turns = [], [0], []
    for row in rows:
        entry = acquisition.iloc[row]
        row_waits = [int(el) - WAIT_CORRECTION_CC for el in str(entry["waits_cc"]).split(";")]
        if len(row_waits) != int(entry["pulse_count"]) + 1:
            raise ValueError(f"Row {entry['row_id']}: expected one wait per pi pulse, plus one")
        waits.extend(row_waits)
        offsets.append(len(waits))
        analysis_turns.append(AXIS_TURNS[analysis_axis[entry["row_id"]]])
    if np.any(np.less(waits, 4)):
        raise RuntimeError("Wait shorter than 4 clock cycles")
    prep_turns = AXIS_TURNS[prep_axes.pop()]
    pi_turns = AXIS_TURNS[pi_axes.pop()]
    return waits, offsets, analysis_turns, prep_turns, pi_turns


def macro_rotation(uwave_ind_list, pulse_macro, turns):
    """Play pulse_macro (pi or pi/2) about the axis at `turns` (0 = +X, 0.25 = +Y)."""
    if isinstance(turns, (int, float)) and turns == 0:
        pulse_macro(uwave_ind_list)
        return
    # Check: this must return the IQ microwave element of each uwave_ind in your config
    elements = [seq_utils.get_sig_gen_element(uwave_ind) for uwave_ind in uwave_ind_list]
    for element in elements:
        qua.frame_rotation_2pi(turns, element)
    pulse_macro(uwave_ind_list)
    qua.reset_frame(*elements)


def get_seq(base_scc_seq_args, step_vals, timing_dir, rows, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    waits, offsets, analysis_turns, prep_turns, pi_turns = load_rqs_rows(timing_dir, rows)

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        step_val = qua.declare(int)
        waits_qua = qua.declare(int, value=waits)
        offsets_qua = qua.declare(int, value=offsets)
        analysis_turns_qua = qua.declare(qua.fixed, value=analysis_turns)
        row = qua.declare(int)
        wait_ind = qua.declare(int)
        last_ind = qua.declare(int)
        turns = qua.declare(qua.fixed)

        def uwave_macro_sig(uwave_ind_list, step_val):
            # step = 2 * (index in rows) + phase; phase 1 = analysis pi/2 shifted by 180 deg
            qua.assign(row, step_val >> 1)
            qua.assign(last_ind, offsets_qua[row + 1] - 1)
            qua.assign(turns, analysis_turns_qua[row])
            with qua.if_((step_val & 1) == 1):
                qua.assign(turns, turns + 0.5)
            qua.align()
            macro_rotation(uwave_ind_list, seq_utils.macro_pi_on_2_pulse, prep_turns)
            with qua.for_(wait_ind, offsets_qua[row], wait_ind < last_ind, wait_ind + 1):
                qua.wait(waits_qua[wait_ind])
                macro_rotation(uwave_ind_list, seq_utils.macro_pi_pulse, pi_turns)
            qua.wait(waits_qua[last_ind])
            macro_rotation(uwave_ind_list, seq_utils.macro_pi_on_2_pulse, turns)
            qua.wait(buffer)

        with qua.for_each_(step_val, step_vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                step_val,
                num_reps,
            )
    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    # Simulate row 0 in both phases, then row 1, and compare the pi-pulse starts in the
    # plot with the converter's times printed here
    timing_dir = r"C:\Users\saroj\Github\dioptric\RQS_toolkit\RQS_toolkit\outputs\rqs_phase_locked_opx_timing"
    rows = [0, 1]

    try:
        acquisition = pd.read_csv(Path(timing_dir) / "opx_ready_acquisition.csv")
        for row in rows:
            print(
                acquisition["row_id"][row],
                "pi starts (ns after the first pi/2 starts):",
                acquisition["pi_starts_ns_from_first_pi2"][row],
            )
        seq, seq_ret_vals = get_seq(
            [
                [[109.062, 107.003], [110.183, 105.383]],
                [1000, 1000],
                [1.0, 1.0],
                [[73.477, 72.33], [74.328, 70.992]],
                [168, 184],
                [1.0, 1.0],
                [False, False, False],
                [1],
            ],
            [0, 1, 2],
            timing_dir,
            rows,
            1,
        )

        sim_config = SimulationConfig(duration=int(600e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
