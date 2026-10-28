# -*- coding: utf-8 -*-
"""
Widefield crosstalk check
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence
from utils import common
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


def get_seq(
    base_scc_seq_args,
    laser_name,
    crosstalk_coords_list,
    num_reps=1,
):
    ### Non-QUA stuff

    (
        pol_coords_list,
        scc_coords_list,
        scc_duration_list,
        spin_flip_ind_list,
        uwave_ind_list,
    ) = base_scc_seq_args
    reference = True
    crosstalk_x_coords_list = [int(el[0] * 10**6) for el in crosstalk_coords_list]
    crosstalk_y_coords_list = [int(el[1] * 10**6) for el in crosstalk_coords_list]
    uwave_ind_list = base_scc_seq_args[-1]

    def uwave_macro_sig(uwave_ind_list, step_val):
        seq_utils.macro_pi_pulse(uwave_ind_list)

    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]

    if num_reps is None:
        num_reps = 1

    # Construct the list of experiments to run
    uwave_macro = [uwave_macro_sig]
    if reference:

        def ref_exp(uwave_ind_list, step_val):
            pass

        uwave_macro.append(ref_exp)
    num_exps_per_rep = len(uwave_macro)
    num_nvs = len(pol_coords_list)

    ### QUA stuff

    with qua.program() as seq:
        seq_utils.init(num_nvs)
        step_val = qua.declare(int)
        crosstalk_x_coord = qua.declare(int)
        crosstalk_y_coord = qua.declare(int)

        def one_exp(exp_ind):
            seq_utils.macro_polarize(pol_coords_list)
            uwave_macro[exp_ind](uwave_ind_list, step_val)

            # Always look at ms=0 counts for the reference
            ref_exp = reference and exp_ind == num_exps_per_rep - 1
            if laser_name == tb.get_physical_laser_name(VirtualLaserKey.ION):
                pulse_name = "scc"
            seq_utils.macro_single_pulse(
                laser_name,
                (crosstalk_x_coord, crosstalk_y_coord),
                pulse_name,
                convert_to_Hz=False,
            )
            seq_utils.macro_scc(scc_coords_list, scc_duration_list)
            seq_utils.macro_charge_state_readout()
            seq_utils.macro_wait_for_trigger()

        def one_rep():
            for exp_ind in range(num_exps_per_rep):
                one_exp(exp_ind)

        def one_step():
            seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)
            seq_utils.macro_pause()

        with qua.for_each_(
            (crosstalk_x_coord, crosstalk_y_coord),
            (crosstalk_x_coords_list, crosstalk_y_coords_list),
        ):
            one_step()

    return seq, []


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[108.89743284830334, 109.37928060155055]],
                [[73.51479542269061, 74.71851917237485]],
                [200],
                [],
                [0, 1],
            ],
            "laser_COBO_638",
            [
                [73.21479542269061, 74.71851917237485],
                [73.51479542269061, 74.71851917237485],
                [73.71479542269061, 74.71851917237485],
            ],
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
