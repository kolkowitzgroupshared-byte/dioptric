# -*- coding: utf-8 -*-
"""Random-timing XY8-8 spectroscopy for 13C, with SCC dual-rail reference.

Each step contains nine *edge-to-edge* wait intervals in OPX clock cycles:
pi/2 - w0 - pi_X - w1 - pi_Y - ... - pi_X - w8 - pi/2.
All patterns have eight pi pulses and therefore the same nominal pulse-error
order. The pulse CENTER times are calculated and saved by the host routine.
This is randomized-filter (RQS-inspired) dephasing spectroscopy, NOT the
noncommuting pooled-spectroscopy algorithm from Kannan's presentation.

The host supplies the random schedules and interleaved XY8 comparison steps.
"""
import numpy as np
import matplotlib.pyplot as plt

from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


PHASES_XY8_DEGREES = (0, 90, 0, 90, 90, 0, 90, 0)
NUM_GAPS = 9


def get_seq(base_scc_seq_args, wait_tables_cc, num_reps=1):
    """Build a single QUA program containing the ordered host-provided steps.

    wait_tables_cc: nine same-length lists, column i = wait after pulse i.
    Every wait must be >= 4 clock cycles (16 ns); QUA waits are in 4-ns cc.
    """
    buffer = seq_utils.get_widefield_operation_buffer()
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        waits = tuple(qua.declare(int) for _ in range(NUM_GAPS))
        uwave_ind_list = base_scc_seq_args[-1]

        def uwave_macro_sig(indices, step_waits):
            qua.align()
            seq_utils.macro_pi_on_2_pulse(indices, phase=0)
            qua.wait(step_waits[0])
            for j, phase in enumerate(PHASES_XY8_DEGREES):
                seq_utils.macro_pi_pulse(indices, phase=phase)
                qua.wait(step_waits[j + 1])
            seq_utils.macro_pi_on_2_pulse(indices, phase=0)
            qua.wait(buffer)

        with qua.for_each_(waits, tuple(wait_tables_cc)):
            base_scc_sequence.macro(
                base_scc_seq_args,
                [uwave_macro_sig],
                waits,
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