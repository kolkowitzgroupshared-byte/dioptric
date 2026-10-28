# -*- coding: utf-8 -*-
"""
Charge state readout after polarization/ionization varying both readout duration and amplitude, no spin manipulation

"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_charge_state_histograms


def get_seq(
    pol_coords_list,
    pol_duration_list,
    pol_amp_list,
    ion_coords_list,
    step_vals,
    optimize_pol_or_readout,
    num_reps,
):
    # Ensure step_vals is a numpy array and convert durations in the first column to clock cycles
    step_vals = np.array(step_vals)
    duration_step_vals = [seq_utils.convert_ns_to_cc(el) for el in step_vals[:, 0]]
    amp_step_vals = step_vals[:, 1]
    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        # Declare override variables
        duration_override = qua.declare(int)
        amp_override = qua.declare(qua.fixed)

        # Determine which variables to override
        pol_duration_override = None
        pol_amp_override = None
        readout_duration_override = None
        readout_amp_override = None
        if optimize_pol_or_readout:
            pol_duration_override = duration_override
            pol_amp_override = amp_override
        else:
            readout_duration_override = duration_override
            readout_amp_override = amp_override

        def one_step():
            base_charge_state_histograms.macro(
                pol_coords_list,
                pol_duration_list,
                pol_amp_list,
                ion_coords_list,
                num_reps,
                pol_duration_override=pol_duration_override,
                pol_amp_override=pol_amp_override,
                readout_duration_override=readout_duration_override,
                readout_amp_override=readout_amp_override,
            )

        with qua.for_each_(duration_override, duration_step_vals):
            with qua.for_each_(amp_override, amp_step_vals):
                one_step()

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq_args = [
            [
                [108.793, 106.879],
                [109.929, 105.244],
                [110.15, 108.507],
                [107.701, 104.991],
                [106.534, 106.305],
            ],
            [1000, 1000, 1000, 1000, 1000],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [
                [73.495, 72.409],
                [74.355, 71.074],
                [74.614, 73.732],
                [72.581, 70.875],
                [71.689, 71.95],
            ],
            [
                [3.6e07, 1.2e00],
                [4.8e07, 8.0e-01],
                [1.2e07, 1.2e00],
                [1.2e07, 1.1e00],
                [6.0e07, 9.0e-01],
                [4.8e07, 1.1e00],
                [6.0e07, 1.2e00],
                [2.4e07, 1.2e00],
                [3.6e07, 1.0e00],
                [6.0e07, 1.0e00],
                [4.8e07, 9.0e-01],
                [3.6e07, 8.0e-01],
                [2.4e07, 9.0e-01],
                [2.4e07, 8.0e-01],
                [1.2e07, 9.0e-01],
                [6.0e07, 1.1e00],
                [4.8e07, 1.2e00],
                [1.2e07, 8.0e-01],
                [2.4e07, 1.1e00],
                [3.6e07, 9.0e-01],
                [6.0e07, 8.0e-01],
                [4.8e07, 1.0e00],
                [2.4e07, 1.0e00],
                [1.2e07, 1.0e00],
                [3.6e07, 1.1e00],
            ],
            False,
        ]
        seq, seq_ret_vals = get_seq(*seq_args, 5)

        sim_config = SimulationConfig(duration=int(150e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
        plt.show(block=True)
