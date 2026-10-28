# -*- coding: utf-8 -*-
"""
Widefield correlation measurement
"""


import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


def get_seq(args, num_reps):
    (pol_coords_list, ion_coords_list, tau_ns, anticorrelation_inds) = args

    tau = seq_utils.convert_ns_to_cc(tau_ns, allow_zero=True)
    short_wait = seq_utils.convert_ns_to_cc(100)
    buffer = seq_utils.get_widefield_operation_buffer()
    sig_gen_el = seq_utils.get_sig_gen_element()
    i_el, q_el = seq_utils.get_iq_mod_elements()

    anticorrelation = isinstance(anticorrelation_inds, list)
    if anticorrelation:
        anti_pol_coords_list = [pol_coords_list[ind] for ind in anticorrelation_inds]

    def setup_macro():
        random = qua.Random()
        rand_phase = qua.declare(qua.fixed)
        return random, rand_phase

    def uwave_macro(random, rand_phase):
        if anticorrelation:
            qua.play("pi_pulse", i_el)
            qua.play("pi_pulse", sig_gen_el)
            qua.align()
            seq_utils.macro_polarize(anti_pol_coords_list)

        qua.play("pi_on_2_pulse", i_el)
        qua.play("pi_on_2_pulse", sig_gen_el)

        qua.align()

        if tau != 0:
            qua.assign(rand_phase, random.rand_fixed())
            # qua.assign(rand_phase, 0.5)  # MCC
            qua.play(
                "pi_pulse" * qua.amp(qua.Math.cos2pi(rand_phase)), i_el, duration=tau
            )
            qua.play(
                "pi_pulse" * qua.amp(qua.Math.sin2pi(rand_phase)), q_el, duration=tau
            )
            qua.play("pi_pulse", sig_gen_el, duration=tau)
            qua.align()

        qua.play("pi_on_2_pulse", i_el)
        qua.play("pi_on_2_pulse", sig_gen_el)

        qua.wait(buffer, sig_gen_el)

    seq = base_scc_sequence.get_seq(
        pol_coords_list, ion_coords_list, num_reps, uwave_macro, setup_macro=setup_macro
    )

    seq_ret_vals = []
    return seq, seq_ret_vals


if __name__ == "__main__":
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    ip_address = config["DeviceIDs"]["QM_opx_ip"]
    qmm = QuantumMachinesManager(host=ip_address)
    opx = qmm.open_qm(opx_config)

    try:
        args = [
            [
                [111.4994186339929, 108.79019926783882],
                [110.7254186339929, 109.27119926783882],
                [111.4994186339929, 108.79019926783882],
                [110.7254186339929, 109.27119926783882],
            ],
            [
                [75.60717320911249, 74.33558456443815],
                [74.88417320911249, 74.52458456443814],
                [75.60717320911249, 74.33558456443815],
                [74.88417320911249, 74.52458456443814],
            ],
            64.0,
            [1, 2],
        ]
        seq, seq_ret_vals = get_seq(args, 5)

        sim_config = SimulationConfig(duration=int(800e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
