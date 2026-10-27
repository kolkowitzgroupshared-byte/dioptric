"""Cornell arbitrary SU(2): one resonant gate pulse with a tracked Z frame.

In a coherent logical frame, chronological operations are

    Rz(gamma) [virtual] -> Ry(beta) [one physical IQ pulse]
    -> Rz(alpha) [virtual] -> logical Rx(pi/2) projection

for U = Rz(alpha) Ry(beta) Rz(gamma), where physical IQ phases are
shifted by minus the accumulated frame.

The physical implementation is:
    standard prep(+Z / +X / +Y);
    one resonant gate pulse at (90 - gamma);
    one frame-corrected projection at -(alpha + gamma);
    SCC readout.

Only the *Cornell gate* uses one physical pulse: +X/+Y preparations and
the final analysis rotation remain physical pi/2 pulses.

No hardware interaction occurs here; only manifests and expected responses
are constructed. Original 'projection' and 'virtual_z' modes are untouched.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from utils.single_qubit_gate_compiler import source_rotation
from utils.virtual_z_gate_compiler import (
    CLOCK_NS, PHASE_GRID_DEG, MAX_ROT_ERROR_RAD,
    compile_virtual_z_manifest,
)

STANDARD_PREP_PHASES_DEG = {"+Z": None, "+X": 90, "+Y": 180}
INPUT_VECTORS = (
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
)


def _rxy(phase_deg, angle_rad):
    rad = math.radians(phase_deg)
    return Rotation.from_rotvec(
        angle_rad * np.array([math.cos(rad), math.sin(rad), 0.0])
    )


def _rz(phase_deg):
    return Rotation.from_rotvec([0.0, 0.0, math.radians(phase_deg)])


def compile_virtual_z_frame_manifest(baseline, pi_ns):
    """Compile the same logical Cornell gates with unchanged input preparation.

    Use the existing virtual-Z compiler's optimal 1-pulse angle/phase/timing
    search, but shift the *gate and readout phase*, not preparation phases.

    This maintains the historical virtual-Z compilation quality and timing.
    It also explicitly represents both (quantized) logical frame changes.
    """
    # Keep original all-projected behavior for the separate frame-tracked mode.
    old = compile_virtual_z_manifest(baseline, pi_ns, project_plus_z=True)
    manifest = dict(old)
    compiled = []
    phases = {0, 90, 180, 270}
    pi_ns = float(pi_ns)

    for prior in old["gates"]:
        gate = dict(prior)
        pulse_list = [dict(p) for p in prior["pulses"]]
        delta = int(prior["virtual_z_phase_compiled_deg"]) % 360
        proj_phase = (-delta) % 360

        if len(pulse_list) > 1:
            raise ValueError("One-gate-pulse contract violated")
        if pulse_list:
            old_phase = int(pulse_list[0]["phase_deg"])
            physical_phase = (old_phase - delta) % 360
            pulse_list[0]["phase_deg"] = physical_phase
            rotation = _rxy(
                physical_phase,
                float(pulse_list[0]["actual_angle_rad"]),
            )
            # Frame angle gamma sets the transverse microwave phase for a
            # nominal +Y operation: phi_lab = (90 - gamma) modulo 360.
            gamma = (90 - physical_phase) % 360
        else:
            rotation = Rotation.identity()
            gamma = 0

        alpha = (delta - gamma) % 360
        logical = _rz(delta) * rotation
        target = source_rotation(gate)
        projection = _rxy(0, math.pi / 2)
        corrected_projection = _rxy(proj_phase, math.pi / 2)
        ideal = np.array([
            projection.apply(target.apply(inp))[2] for inp in INPUT_VECTORS
        ])
        actual = np.array([
            corrected_projection.apply(rotation.apply(inp))[2]
            for inp in INPUT_VECTORS
        ])
        err = float((target.inv() * logical).magnitude())
        response_error = float(np.linalg.norm(actual - ideal))

        # Assert conjugation/frame bookkeeping and full-gate equivalence.
        old_rotation = Rotation.identity()
        if prior["pulses"]:
            old_rotation = _rxy(
                prior["pulses"][0]["phase_deg"],
                float(prior["pulses"][0]["actual_angle_rad"]),
            )
        old_logical = old_rotation * _rz(delta)
        if (old_logical.inv() * logical).magnitude() > 2e-10:
            raise AssertionError("Software-frame identity failed")
        if not np.allclose(actual, prior["predicted_projected_z"], atol=1e-11):
            raise AssertionError("Frame-corrected projected-Z prediction changed")
        if abs(err - prior["rotation_error_rad"]) > 2e-10:
            raise AssertionError("Frame transformation changed logical SO3 error")

        phases.add(proj_phase)
        phases.update(p["phase_deg"] for p in pulse_list)
        gate.update({
            "decomposition": "virtual_Z_frame_tracked",
            "readout_protocol": "virtual_z_frame_projection",
            "preparation_phases_deg": dict(STANDARD_PREP_PHASES_DEG),
            "projection_phase_deg": proj_phase,
            "logical_projection_phase_deg": 0,
            "frame_before_gate_deg": int(gamma),
            "frame_after_gate_update_deg": int(alpha),
            "frame_final_deg": int(delta),
            "frame_initial_deg": 0,
            "virtual_z_phase_compiled_deg": delta,
            "pulses": pulse_list,
            "rotation_error_rad": err,
            "projected_z_response_error_compiler": response_error,
            "expected_projected_z": ideal.tolist(),
            "predicted_projected_z": actual.tolist(),
            "predicted_actual_bloch": logical.apply(INPUT_VECTORS[0]).tolist(),
        })
        compiled.append(gate)

    if 360 % PHASE_GRID_DEG:
        raise ValueError("IQ phase grid must divide 360 degrees")
    if any(p % PHASE_GRID_DEG for p in phases):
        raise ValueError("Compiled IQ phase lies off configured grid")

    manifest.update({
        "gates": compiled,
        "phase_available_deg": sorted(phases),
        "experiment_mode": "virtual_z_frame",
        "readout_protocol": "virtual_z_frame_projection",
        "input_state_order": ["+Z", "+X", "+Y"],
        "projection_axis": "+X (logical); physical phase frame-corrected",
        "projection_phase_deg": None,  # Individual phase in each gate.
        "projected_observable": (
            "logical SCC Z after Rx(+pi/2) with frame-corrected physical phase"
        ),
        "virtual_z_method": (
            "same physical preparations as projection, one physical transverse "
            "gate pulse, logical Z frame shifts and phase-adjusted projection"
        ),
        "iq_phase_grid_deg": PHASE_GRID_DEG,
        "max_rotation_error_rad": max(g["rotation_error_rad"] for g in compiled),
        "max_projected_z_compiler_error": max(
            g["projected_z_response_error_compiler"] for g in compiled
        ),
    })
    manifest["hardware_warning"] = (
        manifest["max_rotation_error_rad"] > MAX_ROT_ERROR_RAD
    )
    return manifest
