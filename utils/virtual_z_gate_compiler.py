"""Offline virtual-Z compiler for Purcell Cornell Experiment 0.

Virtual Z is encoded in the preparation pulse phases for known +Z/+X/+Y
initial states. One transverse IQ microwave gate pulse follows. By default
only +X/+Y receive a final +X pi/2; +Z is read out directly in Z.
The historic all-projected protocol remains available for frame-mode compilation. This reproduces the requested gate's response
for these known inputs, up to OPX phase and timing quantization.

It is NOT a general-purpose standalone physical SU(2) implementation on an
unknown input unless the tracked frame is carried into surrounding operations.
No OPX configuration is changed here.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation
from utils.single_qubit_gate_compiler import source_rotation

PHASE_GRID_DEG = 1
CLOCK_NS = 4
MIN_PULSE_NS = 16
MAX_ROT_ERROR_RAD = 0.15
INPUTS = (
    ("+Z", np.array([0.0, 0.0, 1.0]), None),
    ("+X", np.array([1.0, 0.0, 0.0]), 90),
    ("+Y", np.array([0.0, 1.0, 0.0]), 180),
)
PROJECTION = Rotation.from_rotvec([math.pi / 2, 0, 0])


def _phase_candidates(angle_deg):
    """Nearest available 9-degree waveforms, plus neighbors."""
    center = int(round(angle_deg / PHASE_GRID_DEG)) * PHASE_GRID_DEG
    return sorted({(center + i * PHASE_GRID_DEG) % 360 for i in (-2, -1, 0, 1, 2)})


def _duration_candidates(angle_rad, pi_ns):
    center_cc = int(round(abs(angle_rad) * pi_ns / math.pi / CLOCK_NS))
    return [
        cc for cc in range(max(MIN_PULSE_NS // CLOCK_NS, center_cc - 2),
                           center_cc + 3)
        if cc * CLOCK_NS >= MIN_PULSE_NS
    ]


def compile_virtual_z_gate(gate, pi_ns, *, project_plus_z=False):
    """Compile one Cornell Rz(a)Ry(b)Rz(c) with quantized physical I/Q phases.

    SO(3) Euler convention uses matrices acting right-to-left. Our protocol
    is prep(+Z/+X/+Y) with a phase shift delta=a+c, followed by ONE physical
    transverse rotation with phase a+90 degrees. The final Rx(+pi/2) is phase 0 for +X/+Y; +Z can use direct Z.

    The alternative longer 2pi-beta pulse around the opposite axis is only
    considered when the short beta pulse is too short for the OPX.
    """
    target = source_rotation(gate)
    a, beta, c = target.as_euler("ZYZ")
    delta_deg = math.degrees(a + c)
    transverse_deg = math.degrees(a) + 90
    pi_ns = float(pi_ns)
    if pi_ns < 32 or pi_ns % CLOCK_NS:
        raise ValueError("pi_ns must be >= 32 and divisible by 4")

    candidates = []
    rotation_options = [(beta, transverse_deg)]
    if abs(beta) < MIN_PULSE_NS * math.pi / pi_ns:
        # R_{phi+pi}(2pi-beta) is the same SO(3) rotation as R_phi(beta).
        rotation_options = [(2 * math.pi - beta, transverse_deg + 180)]

    for actual_delta_deg in _phase_candidates(delta_deg):
        rz = Rotation.from_rotvec([0, 0, math.radians(actual_delta_deg)])
        prep_phases = {
            "+Z": None,
            "+X": int((90 + actual_delta_deg) % 360),
            "+Y": int((180 + actual_delta_deg) % 360),
        }
        for desired_beta, desired_phase in rotation_options:
            if abs(desired_beta) < 1e-11:
                choices = [(None, 0, 0.0)]
            else:
                choices = [
                    (int(phase), cc * CLOCK_NS,
                     cc * CLOCK_NS * math.pi / pi_ns)
                    for phase in _phase_candidates(desired_phase)
                    for cc in _duration_candidates(desired_beta, pi_ns)
                ]
            for phase, duration_ns, rotation_rad in choices:
                if phase is None:
                    drive = Rotation.identity()
                else:
                    phi = math.radians(phase)
                    drive = Rotation.from_rotvec(
                        rotation_rad * np.array([math.cos(phi), math.sin(phi), 0])
                    )
                logical = drive * rz
                error_rad = float((target.inv() * logical).magnitude())
                predicted_z = [
                    float((PROJECTION.apply(logical.apply(state))
                           if project_plus_z or label != "+Z"
                           else logical.apply(state))[2])
                    for label, state, _ in INPUTS
                ]
                ideal_z = [
                    float((PROJECTION.apply(target.apply(state))
                           if project_plus_z or label != "+Z"
                           else target.apply(state))[2])
                    for label, state, _ in INPUTS
                ]
                response_error = float(np.linalg.norm(
                    np.asarray(predicted_z) - np.asarray(ideal_z)
                ))
                # Minimize full-gate SO(3) error; prefer shorter duration
                # within the same ~0.01 rad error bin.
                cost = (round(error_rad, 2), duration_ns, error_rad)
                candidates.append((cost, phase, duration_ns,
                                   rotation_rad, prep_phases,
                                   actual_delta_deg, logical,
                                   predicted_z, ideal_z, error_rad,
                                   response_error))
    if not candidates:
        raise ValueError(f"No virtual-Z compilation for gate {gate['gate']}")
    (cost, phase, duration_ns, angle_rad, prep_phases, actual_delta,
     logical, predicted, ideal, error_rad, response_error) = min(
         candidates, key=lambda item: item[0]
     )
    result = dict(gate)
    result["baseline_three_pulse"] = gate["pulses"]
    result["baseline_rotation_error_rad"] = gate["rotation_error_rad"]
    result["decomposition"] = "virtual_Z_phase_tracked"
    result["readout_protocol"] = (
        "virtual_z_projection" if project_plus_z
        else "virtual_z_mixed_projection"
    )
    result["final_projection_by_input"] = {
        "+Z": bool(project_plus_z), "+X": True, "+Y": True
    }
    result["projection_phase_deg"] = 0
    result["preparation_phases_deg"] = prep_phases
    result["virtual_z_phase_ideal_deg"] = float(delta_deg % 360)
    result["virtual_z_phase_compiled_deg"] = int(actual_delta)
    result["transverse_phase_ideal_deg"] = float(transverse_deg % 360)
    result["pulses"] = [] if phase is None else [{
        "axis": "XY",
        "phase_deg": int(phase),
        "duration_ns": int(duration_ns),
        "duration_cc": int(duration_ns // CLOCK_NS),
        "requested_angle_rad": float(beta),
        "actual_angle_rad": float(angle_rad),
    }]
    result["duration_total_ns"] = int(duration_ns)
    result["rotation_error_rad"] = float(error_rad)
    result["projected_z_response_error_compiler"] = response_error
    result["expected_projected_z"] = ideal
    result["predicted_projected_z"] = predicted
    result["predicted_actual_bloch"] = logical.apply([0, 0, 1]).tolist()
    return result


def compile_virtual_z_manifest(baseline, pi_ns, *, project_plus_z=False):
    """Return a new manifest, without mutating the original three-pulse one."""
    manifest = dict(baseline)
    manifest["gates"] = [
        compile_virtual_z_gate(g, pi_ns, project_plus_z=project_plus_z) for g in baseline["gates"]
    ]
    all_phases = {0, 90, 180, 270}
    for g in manifest["gates"]:
        all_phases.update(
            p["phase_deg"] for p in g["pulses"]
        )
        all_phases.update(
            p for p in g["preparation_phases_deg"].values() if p is not None
        )
    manifest["phase_available_deg"] = sorted(all_phases)
    manifest["experiment_mode"] = "virtual_z"
    manifest["readout_protocol"] = (
        "virtual_z_projection" if project_plus_z
        else "virtual_z_mixed_projection"
    )
    manifest["final_projection_by_input"] = {
        "+Z": bool(project_plus_z), "+X": True, "+Y": True
    }
    manifest["input_state_order"] = ["+Z", "+X", "+Y"]
    manifest["projection_axis"] = "+X"
    manifest["projection_phase_deg"] = 0
    manifest["projected_observable"] = (
        "SCC Z after fixed Rx(+pi/2); pre-projection Y for +Z/+X/+Y"
        if project_plus_z else
        "SCC direct Z for +Z; pre-projection Y for +X/+Y after fixed Rx(+pi/2)"
    )
    manifest["virtual_z_method"] = (
        "preparation phase shifts plus one transverse microwave gate pulse"
    )
    manifest["iq_phase_grid_deg"] = PHASE_GRID_DEG
    manifest["max_rotation_error_rad"] = max(
        gate["rotation_error_rad"] for gate in manifest["gates"]
    )
    manifest["max_projected_z_compiler_error"] = max(
        gate["projected_z_response_error_compiler"] for gate in manifest["gates"]
    )
    manifest["hardware_warning"] = (
        manifest["max_rotation_error_rad"] > MAX_ROT_ERROR_RAD
    )
    return manifest
