"""
Widefield Ramsey Phase Scan Test

Created on March 25th, 2025
"""

import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(base_scc_seq_args, seq_names, num_reps=1):
    buffer = seq_utils.get_widefield_operation_buffer()
    bootstrap_dict = {
        "pi_2_X": [("pi/2", 0)],
        "pi_2_Y": [("pi/2", 90)],
        "pi_2_X_pi_X": [("pi/2", 0), ("pi", 0)],
        "pi_2_Y_pi_Y": [("pi/2", 90), ("pi", 90)],
        "pi_Y_pi_2_X": [("pi", 90), ("pi/2", 0)],
        "pi_X_pi_2_Y": [("pi", 0), ("pi/2", 90)],
        "pi_2_Y_pi_2_X": [("pi/2", 90), ("pi/2", 0)],
        "pi_2_X_pi_2_Y": [("pi/2", 0), ("pi/2", 90)],
        "pi_2_X_pi_X_pi_2_Y": [("pi/2", 0), ("pi", 0), ("pi/2", 90)],
        "pi_2_Y_pi_X_pi_2_X": [("pi/2", 90), ("pi", 0), ("pi/2", 0)],
        "pi_2_X_pi_Y_pi_2_Y": [("pi/2", 0), ("pi", 90), ("pi/2", 90)],
        "pi_2_Y_pi_Y_pi_2_X": [("pi/2", 90), ("pi", 90), ("pi/2", 0)],
        # created for test not a part of the original bootstrap sequence
        "pi_2_Y_pi_X_pi_2_Y": [("pi/2", 90), ("pi", 0), ("pi/2", 90)],
    }

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()
        uwave_macro_list = []
        print(seq_names)
        for seq_name in seq_names:
            if seq_name not in bootstrap_dict:
                raise ValueError(f"Unknown sequence name: {seq_name}")

            pulse_list = bootstrap_dict[seq_name]

            def make_sig_macro(pulses):
                def macro_fn(
                    uwave_ind_list, step_val, pulses=pulses
                ):  # bind pulse sequence
                    qua.align()
                    with qua.strict_timing_():
                        for kind, phase in pulses:
                            if kind == "pi/2":
                                seq_utils.macro_pi_on_2_pulse(
                                    uwave_ind_list, phase=phase
                                )
                            elif kind == "pi":
                                seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)
                    qua.wait(buffer)

                return macro_fn

            sig_macro = make_sig_macro(pulse_list)
            uwave_macro_list.append(sig_macro)

        print(len(uwave_macro_list))

        base_scc_sequence.macro(
            base_scc_seq_args,
            uwave_macro_list,
            num_reps=num_reps,
            reference=True,
        )

    return seq, []


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 10e3

    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, seq_ret_vals = get_seq(
            [
                [[107.715, 107.724], [107.438, 105.983]],
                [132, 132],
                [1.0, 1.0],
                [[72.439, 73.236], [72.175, 71.822]],
                [40, 88],
                [1.0, 1.0],
                [False, False],
                [1],
            ],
            [
                "pi_2_Y_pi_X_pi_2_Y",  # created for test
                "pi_2_X",
                "pi_2_Y",
                "pi_2_X_pi_X",
                "pi_2_Y_pi_Y",
                "pi_Y_pi_2_X",
                "pi_X_pi_2_Y",
                "pi_2_Y_pi_2_X",
                "pi_2_X_pi_2_Y",
                "pi_2_X_pi_X_pi_2_Y",
                "pi_2_Y_pi_X_pi_2_X",
                "pi_2_X_pi_Y_pi_2_Y",
                "pi_2_Y_pi_Y_pi_2_X",
            ],
            1,
        )

        sim_config = SimulationConfig(duration=int(50e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception:
        traceback.print_exc()
    finally:
        qmm.close_all_quantum_machines()
