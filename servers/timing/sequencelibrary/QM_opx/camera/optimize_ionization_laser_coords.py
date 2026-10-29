# -*- coding: utf-8 -*-
"""
Scanning illumination and widefield collection

"""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, qua
from qm.simulate import SimulationConfig

from servers.timing.sequencelibrary.QM_opx import seq_utils
from utils import common
from utils import tool_belt as tb
from utils.constants import VirtualLaserKey


def get_seq(pol_coords, ion_coords, num_reps):
    if num_reps is None:
        num_reps = 1

    green_laser = tb.get_physical_laser_name(VirtualLaserKey.CHARGE_POL)
    red_laser = tb.get_physical_laser_name(VirtualLaserKey.ION)

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods(
            [green_laser, red_laser], aod_suffices=["charge_pol", "opti"]
        )

        def one_rep(rep_ind=None):
            # Charge polarization with green, spin polarization with yellow
            seq_utils.macro_polarize([pol_coords], spin_pol=False)

            # Ionization
            seq_utils.macro_ionize([ion_coords])

            # Readout
            seq_utils.macro_charge_state_readout()

            seq_utils.macro_wait_for_trigger()

        seq_utils.handle_reps(one_rep, num_reps, wait_for_trigger=False)

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
        seq, seq_ret_vals = get_seq([111.202, 109.801], [75, 75], 5)

        sim_config = SimulationConfig(duration=int(200e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
