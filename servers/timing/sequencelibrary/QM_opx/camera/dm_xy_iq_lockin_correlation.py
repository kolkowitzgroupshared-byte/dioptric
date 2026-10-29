# -*- coding: utf-8 -*-
"""
DM lock-in search sequence (widefield) using XY4-N and 4-shot I/Q phase cycling.

Experiments per rep (in this order):
  0: I+  (final pi/2 phase = +X)
  1: I-  (final pi/2 phase = -X)
  2: Q+  (final pi/2 phase = +Y)
  3: Q-  (final pi/2 phase = -Y)
  4: optional reference (MW off)

Input args (encoded by tb.encode_seq_args):
  base_scc_seq_args, tau_ns, n_xy4_blocks, include_ref

tau_ns is the Hahn-tau convention you already use:
  - Hahn total evolution = 2*tau
  - lock-in center freq ~ 1/(2*tau)

This implementation uses XY4 with pulses separated by 2*tau:
  (pi/2)_X - tau - [pi_X - 2tau - pi_Y - 2tau - pi_X - 2tau - pi_Y] - tau - (pi/2)_phi
Repeated n_xy4_blocks times with continuous 2tau spacing between pi pulses.
"""

import numpy as np
from qm import qua
import utils.tool_belt as tb
import utils.common as common
from servers.timing.sequencelibrary.QM_opx import seq_utils
from servers.timing.sequencelibrary.QM_opx.camera import base_scc_sequence


# -----------------
# Phase conventions
# -----------------
PH_X   = 0
PH_Y   = 90
PH_mX  = 180
PH_mY  = 270


def _echo_core_npi(uwave_ind_list, tau_cc, final_phase_deg, n_pi=1, pi_phases_deg=None):
    """
    (pi/2)_X  - tau - [pi ... spaced by 2*tau] - tau - (pi/2)_final

    NOTE: total free evolution time = 2 * n_pi * tau
          (n_pi=1 -> 2*tau; n_pi=2 -> 4*tau)
    """
    if n_pi < 0:
        raise ValueError("n_pi must be >= 0")

    two_tau_cc = 2 * tau_cc

    if pi_phases_deg is None:
        if n_pi == 0:
            pi_phases_deg = []
        elif n_pi == 1:
            pi_phases_deg = [PH_X]             # Hahn
        elif n_pi == 2:
            pi_phases_deg = [PH_X, PH_Y]       # simple XY2-like
        else:
            pi_phases_deg = [(PH_X if (k % 2 == 0) else PH_Y) for k in range(n_pi)]
    else:
        if len(pi_phases_deg) != n_pi:
            raise ValueError("len(pi_phases_deg) must equal n_pi")

    seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=PH_X)
    qua.wait(tau_cc)

    for k, phase in enumerate(pi_phases_deg):
        seq_utils.macro_pi_pulse(uwave_ind_list, phase=phase)
        if k < n_pi - 1:
            qua.wait(two_tau_cc)
        else:
            qua.wait(tau_cc)

    seq_utils.macro_pi_on_2_pulse(uwave_ind_list, phase=final_phase_deg)
    return False


def get_seq(base_scc_seq_args, tau_ns, n_pi=1, num_reps=1):
    """
    If keep_total_evolution_fixed=True:
      - tau_ns is interpreted as your *Hahn tau* (i.e., total evolution = 2*tau_ns)
      - For n_pi>1 we shrink the per-endcap tau so total evolution stays 2*tau_ns:
            tau_cc = tau_ns / n_pi
    If keep_total_evolution_fixed=False:
      - tau_ns is the per-endcap tau used in the diagram; total evolution = 2*n_pi*tau_ns
    """

    tau_cc = seq_utils.convert_ns_to_cc(tau_ns)

    with qua.program() as seq:
        seq_utils.init()
        seq_utils.macro_run_aods()

        def uwave_Ip(uwave_ind_list, step_val):
            return _echo_core_npi(uwave_ind_list, tau_cc, PH_X,  n_pi=n_pi)

        def uwave_Im(uwave_ind_list, step_val):
            return _echo_core_npi(uwave_ind_list, tau_cc, PH_mX, n_pi=n_pi)

        def uwave_Qp(uwave_ind_list, step_val):
            return _echo_core_npi(uwave_ind_list, tau_cc, PH_Y,  n_pi=n_pi)

        def uwave_Qm(uwave_ind_list, step_val):
            return _echo_core_npi(uwave_ind_list, tau_cc, PH_mY, n_pi=n_pi)
        
        def uwave_ref(uwave_ind_list, step_val):
            pass

        uwave_macros = [uwave_Ip, uwave_Im, uwave_Qp, uwave_Qm, uwave_ref]

        base_scc_sequence.macro(
            base_scc_seq_args,
            uwave_macros,
            num_reps=num_reps,
            reference=False,
        )

    return seq, []

if __name__ == "__main__":
    # Local simulation hook (optional)
    import matplotlib.pyplot as plt
    from qm import QuantumMachinesManager
    from qm.simulate import SimulationConfig

    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config
    tb.set_delays_to_zero(opx_config)
    opx_config["pulses"]["yellow_spin_pol"]["length"] = 1e3
    
    qm_opx_args = config["DeviceIDs"]["QM_opx_args"]
    qmm = QuantumMachinesManager(**qm_opx_args)
    opx = qmm.open_qm(opx_config)

    try:
        seq, _ = get_seq(
            [
                [[109.114, 107.084], [110.468, 108.724]],
                [1000, 1000],
                [1.0, 1.0],
                [[73.686, 72.605], [74.759, 73.921]],
                [116, 108],
                [1.0, 1.0],
                [False, False],
                [0, 1],
            ],
            tau_ns=15e3,
            n_pi=1,
            num_reps=1,
        )
        sim_config = SimulationConfig(duration=int(100e3 / 4))
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        raise exc
    finally:
        qmm.close_all_quantum_machines()
