# -*- coding: utf-8 -*-
"""
Scanning illumination and widefield collection

Created on October 13th, 2023

@author: mccambria
"""


import matplotlib.pyplot as plt
import numpy
from qm import QuantumMachinesManager, generate_qua_script, qua
from qm.simulate import SimulationConfig

import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils


def get_seq(
    readout_duration_ns,
    readout_laser,
    #
    do_polarize,
    pol_laser,
    pol_coords,
    pol_duration_ns,
    #
    do_ionize,
    ion_laser,
    ion_coords,
    ion_duration_ns,
    #
    num_reps,
):
    if num_reps == None:
        num_reps = 1

    readout_laser_el = seq_utils.get_laser_mod_element(readout_laser)
    camera_el = f"do_camera_trigger"

    pol_laser_el = f"do_{pol_laser}_dm"
    pol_x_el = f"ao_{pol_laser}_x"
    pol_y_el = f"ao_{pol_laser}_y"

    ion_laser_el = f"do_{ion_laser}_dm"
    ion_x_el = f"ao_{ion_laser}_x"
    ion_y_el = f"ao_{ion_laser}_y"

    access_time = seq_utils.get_aod_access_time()
    pol_duration = seq_utils.convert_ns_to_cc(pol_duration_ns)
    ion_duration = seq_utils.convert_ns_to_cc(ion_duration_ns)
    default_pulse_duration = seq_utils.get_default_pulse_duration()
    buffer = seq_utils.convert_ns_to_cc(10e3)
    setup_duration = access_time + pol_duration + buffer + ion_duration + buffer
    readout_duration = seq_utils.convert_ns_to_cc(readout_duration_ns)

    with qua.program() as seq:
        pol_x_freq = qua.declare(int, value=round(pol_coords[0] * 10**6))
        pol_y_freq = qua.declare(int, value=round(pol_coords[1] * 10**6))
        qua.update_frequency(pol_x_el, pol_x_freq)
        qua.update_frequency(pol_y_el, pol_y_freq)
        qua.play("aod_cw", pol_x_el)
        qua.play("aod_cw", pol_y_el)

        ion_x_freq = qua.declare(int, value=round(ion_coords[0] * 10**6))
        ion_y_freq = qua.declare(int, value=round(ion_coords[1] * 10**6))
        qua.update_frequency(ion_x_el, ion_x_freq)
        qua.update_frequency(ion_y_el, ion_y_freq)
        qua.play("aod_cw", ion_x_el)
        qua.play("aod_cw", ion_y_el)

        def one_rep():
            # Polarization
            if do_polarize:
                qua.wait(access_time, pol_laser_el)
                qua.play("on", pol_laser_el, duration=pol_duration)

            # Ionization
            if do_ionize:
                qua.wait(access_time + pol_duration + buffer, ion_laser_el)
                qua.play("on", ion_laser_el, duration=ion_duration)

            # Yellow readout
            qua.wait(setup_duration, readout_laser_el)
            qua.wait(setup_duration, camera_el)
            qua.play("charge_readout", readout_laser_el, duration=readout_duration)
            qua.play("on", camera_el)
            qua.align()
            qua.play("off", camera_el)
            qua.align()

        seq_utils.handle_reps(one_rep, num_reps)

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
        # readout_duration, readout_laser, do_polarize, do_ionize, ion_laser, ion_coords, pol_laser, pol_coords,
        args = [
            5e3,
            "laser_OPTO_589",
            True,
            "laser_INTE_520",
            [110.735, 109.668],
            1e3,
            False,
            "laser_COBO_638",
            [74.486, 75.265],
            1e3,
        ]
        seq, seq_ret_vals = get_seq(args, 1)

        sim_config = SimulationConfig(duration=int(50e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
