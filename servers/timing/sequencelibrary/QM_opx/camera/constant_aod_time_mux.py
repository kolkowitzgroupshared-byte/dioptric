#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constant sequence for the QM OPX

Created on April 8th, 2026

@author: saroj chand
"""

import matplotlib.pyplot as plt
import numpy
import qm
from qm import QuantumMachinesManager, generate_qua_script, qua
from qm.simulate import SimulationConfig

import utils.common as common
import utils.kplotlib as kpl
import utils.tool_belt as tb
from servers.timing.sequencelibrary.QM_opx import seq_utils



def _to_hz(coords):
    if coords is None:
        return []
    return [(int(round(x * 1e6)), int(round(y * 1e6))) for x, y in coords]


def get_seq(
    green_coords,
    red_coords,
    green_amp,
    red_amp,
    yellow_amp=None,
    dwell_us=200,
    num_reps=None,
):
    """
    Time-multiplexed multispot AOD display.

    green_coords: list of [fx, fy] in MHz
    red_coords:   list of [fx, fy] in MHz
    green_amp: per-channel voltage amplitude in V
    red_amp:   per-channel voltage amplitude in V
    yellow_amp: optional yellow amplitude in V
    dwell_us: dwell time per spot in microseconds

    This does NOT create a true simultaneous RF sum on one channel.
    It rapidly cycles spot positions so the camera integrates them.
    """

    if num_reps is None:
        num_reps = -1

    if green_amp > 0.5 or red_amp > 0.5 or (yellow_amp is not None and yellow_amp > 0.5):
        raise RuntimeError("Analog voltages must be <= 0.5 V.")

    gcoords = _to_hz(green_coords)
    rcoords = _to_hz(red_coords)

    n_steps = max(len(gcoords), len(rcoords), 1)
    dwell_cycles = max(4, int(round(float(dwell_us) * 250)))  # 1 us = 250 clock cycles

    with qua.program() as seq:
        g_amp = qua.declare(qua.fixed, value=2 * float(green_amp))
        r_amp = qua.declare(qua.fixed, value=2 * float(red_amp))
        y_amp = None
        if yellow_amp is not None:
            y_amp = qua.declare(qua.fixed, value=2 * float(yellow_amp))

        def one_rep():
            for step in range(n_steps):
                # -------------------------
                # Update frequencies
                # -------------------------
                if len(gcoords) > 0:
                    gx, gy = gcoords[step % len(gcoords)]
                    qua.update_frequency("ao3", gx)   # green x
                    qua.update_frequency("ao4", gy)   # green y

                if len(rcoords) > 0:
                    rx, ry = rcoords[step % len(rcoords)]
                    qua.update_frequency("ao2", rx)   # red x
                    qua.update_frequency("ao6", ry)   # red y

                # Sync before each dwell block
                qua.align()

                # -------------------------
                # Turn lasers on for this dwell
                # -------------------------
                if len(gcoords) > 0:
                    qua.play("on", "do4", duration=dwell_cycles)

                if len(rcoords) > 0:
                    qua.play("on", "do1", duration=dwell_cycles)

                # -------------------------
                # Drive AODs during this dwell
                # -------------------------
                if len(gcoords) > 0:
                    qua.play("cw" * qua.amp(g_amp), "ao3", duration=dwell_cycles)
                    qua.play("cw" * qua.amp(g_amp), "ao4", duration=dwell_cycles)

                if len(rcoords) > 0:
                    qua.play("cw" * qua.amp(r_amp), "ao2", duration=dwell_cycles)
                    qua.play("cw" * qua.amp(r_amp), "ao6", duration=dwell_cycles)

                if yellow_amp is not None:
                    qua.play("cw" * qua.amp(y_amp), "ao7", duration=dwell_cycles)

                qua.align()

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
        ret_vals = get_seq([1], [2, 6], [0.19, 0.19], [75, 75])
        seq, seq_ret_vals = ret_vals

        # Serialize to file
        # sourceFile = open('debug2.py', 'w')
        # print(generate_qua_script(seq, opx_config), file=sourceFile)
        # sourceFile.close()

        sim_config = SimulationConfig(duration=10000 // 4)
        sim = opx.simulate(seq, sim_config)
        samples = sim.get_simulated_samples()
        samples.con1.plot()
        plt.show(block=True)

    except Exception as exc:
        print(exc)
    finally:
        qmm.close_all_quantum_machines()
        qmm.close()
