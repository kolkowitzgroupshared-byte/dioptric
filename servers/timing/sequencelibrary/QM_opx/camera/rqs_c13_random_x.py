# -*- coding: utf-8 -*-
"""Random pi_X (and periodic XY8 control) for 13C with optional IQ projection.

The host specifies nine edge-to-edge gap vectors, one binary pulse-axis-mode
vector (1=all-X, 0=periodic XY8), and one readout-basis vector (0 or 90 deg).
All random interrogation pulses are strictly calibrated pi_X. QUA handles
SCC polarization/readout and alternating dual-rail reference as before.
"""
import numpy as np
import matplotlib.pyplot as plt

from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence

def get_seq(base_scc_seq_args, waits_cc, pulse_modes, readout_phases, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        gap_vars = tuple(qua.declare(int) for _ in range(9))
        mode = qua.declare(int)
        basis = qua.declare(int)
        vals = tuple(waits_cc) + (pulse_modes, readout_phases)
        uwave_ind_list = base_scc_seq_args[-1]

        def signal(inds, step_tuple):
            gaps = step_tuple[:9]
            axis_mode = step_tuple[9]
            readout = step_tuple[10]
            qua.align()
            seq_utils.macro_pi_on_2_pulse(inds, phase=0)
            qua.wait(gaps[0])
            for j, xy8_phase in enumerate(AXES_XY8):
                # At every step, the host selects all-X for random masks
                # or the standard XY8 phase cycle for periodic comparisons.
                with qua.if_(axis_mode == 1):
                    seq_utils.macro_pi_pulse(inds, phase=0)
                with qua.else_():
                    seq_utils.macro_pi_pulse(inds, phase=xy8_phase)
                qua.wait(gaps[j + 1])
            with qua.if_(readout == 0):
                seq_utils.macro_pi_on_2_pulse(inds, phase=0)
            with qua.else_():
                seq_utils.macro_pi_on_2_pulse(inds, phase=90)
            qua.wait(buffer)

        with qua.for_each_(gap_vars + (mode, basis), vals):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [signal],
                gap_vars + (mode, basis),
                num_reps,
                reference=True,
            )
    return seq, []

if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 2e3
    
    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[109.062, 107.003], [110.183, 105.383]],
                [1000, 1000],
                [1.0, 1.0],
                [[73.477, 72.33], [74.328, 70.992]],
                [168, 184],
                [1.0, 1.0],
                [False, False, False],
                [0, 1],
            ],
            [
                27200,
                41832,
            ],
            10,
        )

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()