"""Purcell multi-NV Cornell/McMahon Experiment 0.

Runs one Cornell gate at a time:
    gate -> Z, X, Y tomography + dual-rail reference

This keeps each QUA program small enough for the OPX compiler while preserving
the full 10-gate experiment in one final saved data object.

Does not manipulate or overwrite existing OPX configuration files.
"""

from pathlib import Path
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.single_qubit_gate_compiler import (
    compile_gates,
    save_manifest,
    verify_manifest,
    source_rotation,
)
from analysis import arbitrary_gate_tomography as arb_analysis
from utils.virtual_z_gate_compiler import compile_virtual_z_manifest


def _paths():
    root = Path(__file__).resolve().parents[2]
    folder = root / "calibrations" / "purcell" / "arbitrary_gate_tomography"
    return folder / "arbSQR.csv", folder


def _make_calibration_manifest(uwave_ind, pi_ns):
    """Six known pure states using the same IQ chain and SCC readout as Cornell.

    Each row is ONE preparation gate. The existing QUA sequence subsequently
    performs Z, X and Y tomography plus its parity-alternating reference.
    """
    clock_ns = 4
    pi_ns = float(pi_ns)
    if pi_ns < 32 or pi_ns % clock_ns:
        raise ValueError("Calibration requires a pi pulse >=32 ns, divisible by 4 ns")
    half_ns = int(round((pi_ns / 2) / clock_ns)) * clock_ns
    if half_ns < 16:
        raise ValueError("The quantized pi/2 duration is below 16 ns")

    # (state label, preparation IQ phase, pulse duration, target Bloch XYZ)
    specs = [
        ("|0>", None, 0, [0.0, 0.0, 1.0]),
        ("|1>", 0, int(pi_ns), [0.0, 0.0, -1.0]),
        ("+X", 90, half_ns, [1.0, 0.0, 0.0]),
        ("-X", 270, half_ns, [-1.0, 0.0, 0.0]),
        ("+Y", 180, half_ns, [0.0, 1.0, 0.0]),
        ("-Y", 0, half_ns, [0.0, -1.0, 0.0]),
    ]
    axes = {
        0: ("X", np.array([1., 0., 0.]), +1),
        90: ("Y", np.array([0., 1., 0.]), +1),
        180: ("X", np.array([1., 0., 0.]), -1),
        270: ("Y", np.array([0., 1., 0.]), -1),
    }
    gates = []
    for index, (label, phase, duration_ns, expected) in enumerate(specs, start=1):
        if phase is None:
            pulses = []
            predicted = [0.0, 0.0, 1.0]
            rot_error = 0.0
        else:
            axis, direction, sign = axes[phase]
            actual_angle = sign * duration_ns * np.pi / pi_ns
            pulses = [{
                "axis": axis,
                "phase_deg": phase,
                "duration_ns": duration_ns,
                "duration_cc": duration_ns // clock_ns,
                "actual_angle_rad": float(actual_angle),
            }]
            predicted = Rotation.from_rotvec(
                direction * actual_angle
            ).apply([0., 0., 1.]).tolist()
            target_angle = sign * (np.pi if label == "|1>" else np.pi / 2)
            rot_error = float(abs(actual_angle - target_angle))
        gates.append({
            "gate": index,
            "label": label,
            "decomposition": "known-state",
            "pulses": pulses,
            "rotation_error_rad": rot_error,
            "duration_total_ns": duration_ns,
            "expected_bloch_xyz": expected,
            "predicted_actual_bloch": predicted,
        })
    worst_error = max(g["rotation_error_rad"] for g in gates)
    return {
        "format": "cornell-arb-sqr-v1",  # existing verify_manifest hardware contract
        "experiment_mode": "calibration",
        "mw_chain": int(uwave_ind),
        "pi_pulse_ns": pi_ns,
        "pi_on_2_pulse_ns": pi_ns / 2,
        "clock_ns": clock_ns,
        "phase_available_deg": [0, 90, 180, 270],
        "basis_order": ["Z", "X", "Y"],
        "gates": gates,
        "max_rotation_error_rad": worst_error,
        "hardware_warning": worst_error > 0.15,
    }


def _attach_projection_targets(manifest):
    """Derive expected final SCC Z signals for three INITIAL spin states.

    The fixed final Rx(+pi/2) maps pre-projection Y onto the measured Z.
    Therefore each gate yields the Y-row of its SO(3) rotation matrix,
    ordered by initial +Z, +X, +Y; NOT the Cornell XYZ output-state vector.
    """
    inputs = [
        ("+Z", np.array([0., 0., 1.])),
        ("+X", np.array([1., 0., 0.])),
        ("+Y", np.array([0., 1., 0.])),
    ]
    projection = Rotation.from_rotvec([np.pi / 2, 0., 0.])
    axes = {
        "X": np.array([1., 0., 0.]),
        "Y": np.array([0., 1., 0.]),
    }
    manifest["experiment_mode"] = "projection"
    manifest["readout_protocol"] = "fixed_projection"
    manifest["input_state_order"] = [label for label, _ in inputs]
    manifest["projection_axis"] = "+X"
    manifest["projection_phase_deg"] = 0
    manifest["projected_observable"] = "SCC Z after Rx(+pi/2); pre-projection Y"
    for gate in manifest["gates"]:
        ideal = source_rotation(gate)
        compiled = Rotation.identity()
        for pulse in gate["pulses"]:
            compiled = Rotation.from_rotvec(
                axes[pulse["axis"]] * pulse["actual_angle_rad"]
            ) * compiled
        gate["readout_protocol"] = "fixed_projection"
        gate["projection_phase_deg"] = 0
        gate["expected_projected_z"] = [
            float(projection.apply(ideal.apply(state))[2])
            for _, state in inputs
        ]
        gate["predicted_projected_z"] = [
            float(projection.apply(compiled.apply(state))[2])
            for _, state in inputs
        ]
    return manifest


def prepare(uwave_ind=0, mode="cornell"):
    """Precompute Cornell, calibration, projection, or virtual-Z manifest."""
    if mode not in ("cornell", "calibration", "projection", "virtual_z"):
        raise ValueError(
            "mode must be 'cornell', 'calibration', 'projection' or 'virtual_z'"
        )
    csv_path, folder = _paths()
    pi_ns = tb.get_virtual_sig_gen_dict(uwave_ind)["pi_pulse"]

    if mode in ("cornell", "projection", "virtual_z"):
        if not csv_path.exists():
            raise FileNotFoundError(f"Place Cornell CSV at {csv_path}")
        manifest = compile_gates(csv_path, pi_ns, uwave_ind)
        if mode == "projection":
            manifest = _attach_projection_targets(manifest)
            out_path = folder / f"purcell_uwave{uwave_ind}_projection.json"
        elif mode == "virtual_z":
            pi_on_2_ns = tb.get_virtual_sig_gen_dict(uwave_ind)["pi_on_2_pulse"]
            if abs(2.0 * pi_on_2_ns - pi_ns) > 1e-6:
                raise ValueError(
                    "Virtual-Z mode requires calibrated pi_on_2_pulse == pi_pulse/2"
                )
            manifest = compile_virtual_z_manifest(manifest, pi_ns)
            out_path = folder / f"purcell_uwave{uwave_ind}_virtual_z.json"
        else:
            manifest["experiment_mode"] = "cornell"
            out_path = folder / f"purcell_uwave{uwave_ind}.json"
    else:
        manifest = _make_calibration_manifest(uwave_ind, pi_ns)
        out_path = folder / f"purcell_uwave{uwave_ind}_calibration.json"

    save_manifest(manifest, out_path)
    return manifest, out_path


def _check_gate_raw(raw, gate_ind, num_runs):
    """Reject incomplete data returned after a base_routine exception."""
    crash_counter = raw.get("crash_counter", [])
    completed = [
        run_ind
        for run_ind, value in enumerate(crash_counter)
        if value is not None
    ]

    if len(completed) != num_runs:
        raise RuntimeError(
            f"Cornell gate {gate_ind + 1}: only "
            f"{len(completed)}/{num_runs} runs completed. "
            "The underlying QUA/OPX acquisition failed; "
            "see the preceding traceback."
        )

    counts = np.asarray(raw["counts"])

    expected_shape_prefix = (4, len(raw["nv_list"]), num_runs, 1)
    if counts.ndim != 5 or counts.shape[:4] != expected_shape_prefix:
        raise RuntimeError(
            f"Cornell gate {gate_ind + 1}: unexpected counts shape "
            f"{counts.shape}; expected (4, nv, run, 1, rep)."
        )

    return completed


def main(
    nv_list,
    num_reps=8,
    num_runs=5,
    uwave_ind_list=(0,),
    allow_large_error=False,
    show_plots=False,
    mode="cornell",
    save_plots=True,
):
    """Run Cornell or six known-state calibration gates on many NVs.

    mode='cornell' measures 10 arbitrary rotations with XYZ tomography;
    mode='calibration' measures |0>, |1>, +X, -X, +Y, -Y;
    mode='projection' prepares initial +Z/+X/+Y then applies each Cornell
    three-pulse gate and the same final Rx(+pi/2).
    mode='virtual_z' instead phase-shifts input preparation, applies ONE
    transverse gate pulse, and retains the fixed final +X pi/2 projection.
    Both modes acquire only projected Z responses, not XYZ tomography.

    Figures are saved by default without opening blocking plot windows.
    show_plots=True optionally displays them NONBLOCKING.
    Each target gate is compiled and acquired separately.
    Per gate, the OPX program contains 3 signal measurements and one
    parity-alternating dual-rail reference. Legacy signals use Z/X/Y
    tomography; projection signals use initial +Z/+X/+Y, all with
    the same final +X pi/2 projection.

    Final counts shape:
        [gate, experiment=4, NV, run, step=1, repetition]
    """
    
    SEQ_FILE = "arbitrary_gate_tomography.py"
    
    if not nv_list:
        raise ValueError("Empty NV list")

    if len(uwave_ind_list) != 1:
        raise ValueError("One MW chain at a time: use (0,), (1,), or (2,)")

    if num_reps < 4 or num_reps % 2:
        raise ValueError("Use even num_reps >= 4 for alternating dual-rail ref")
    
    chain = int(uwave_ind_list[0])

    manifest, manifest_path = prepare(chain, mode=mode)

    cfg = common.get_config_module()
    pi_ns = tb.get_virtual_sig_gen_dict(chain)["pi_pulse"]

    verify_manifest(
        manifest,
        cfg.opx_config,
        chain,
        pi_ns,
        allow_large_error=allow_large_error,
    )

    print()
    print(f"{mode.upper()} arbitrary-gate/tomography manifest")
    print("--------------------------------")
    for gate in manifest["gates"]:
        print(
            f"Gate {gate['gate']:2d} {gate['decomposition']} : "
            f"{[(p['phase_deg'], p['duration_ns']) for p in gate['pulses']]} ns "
            f"SO3 err={gate['rotation_error_rad']:.4f} rad"
        )
        if mode == "virtual_z":
            print(
                "  virtual Z prep phases:",
                gate["preparation_phases_deg"],
                "projected-Z compiler error:",
                f"{gate['projected_z_response_error_compiler']:.4f}",
            )

    pulse_gen = tb.get_server_pulse_gen()

    gate_raw_list = []

    try:
        for gate_ind, gate in enumerate(manifest["gates"]):
            print()
            print("=" * 70)
            print(
                f"{mode} gate {gate_ind + 1}/{len(manifest['gates'])}: "
                f"{gate.get('label', gate['decomposition'])}"
            )
            print("=" * 70)

            # Bind this gate into this iteration's run callback.
            def run_fn(shuffled_step_inds, gate=gate):
                seq_args = [
                    widefield.get_base_scc_seq_args(
                        nv_list,
                        list(uwave_ind_list),
                    ),
                    [gate],
                ]

                # One gate -> three signal preparations/bases plus reference.
                pulse_gen.stream_load(
                    SEQ_FILE,
                    tb.encode_seq_args(seq_args),
                    num_reps,
                )

            gate_raw = base_routine.main(
                nv_list,
                num_steps=1,
                num_reps=num_reps,
                num_runs=num_runs,
                run_fn=run_fn,
                uwave_ind_list=list(uwave_ind_list),
                num_exps=4,
                load_iq=True,
            )

            _check_gate_raw(
                gate_raw,
                gate_ind,
                num_runs,
            )

            gate_raw_list.append(gate_raw)

            print(
                f"Gate {gate_ind + 1} complete: "
                f"{num_runs}/{num_runs} runs"
            )

        # -------------------------------------------------------------
        # Combine all independently acquired gate blocks.
        # -------------------------------------------------------------
        counts = np.stack(
            [
                np.asarray(gate_raw["counts"])
                for gate_raw in gate_raw_list
            ],
            axis=0,
        )

        pixel_drifts = np.stack(
            [
                np.asarray(gate_raw["pixel_drifts"])
                for gate_raw in gate_raw_list
            ],
            axis=0,
        )

        raw = {
            "nv_list": nv_list,
            "num_gates": len(manifest["gates"]),
            "tomography_mode": mode,
            "num_exps_per_gate": 4,
            "num_steps": 1,
            "num_reps": num_reps,
            "num_runs": num_runs,
            "uwave_ind": list(uwave_ind_list),
            "load_iq": True,
            "counts-units": "photons",
            "counts": counts,
            "pixel_drifts": pixel_drifts,
            "gate_crash_counter": [
                gate_raw["crash_counter"]
                for gate_raw in gate_raw_list
            ],
            "gate_step_ind_master_list": [
                gate_raw["step_ind_master_list"]
                for gate_raw in gate_raw_list
            ],
            "timestamp": dm.get_time_stamp(),
            "cornell_manifest": manifest,
            "cornell_manifest_path": str(manifest_path),
            "experiment": (
                "Known-state tomography calibration"
                if mode == "calibration"
                else "Cornell Experiment 0: virtual-Z phase-tracked projection"
                if mode == "virtual_z"
                else "Cornell Experiment 0: fixed +X pi/2 projection"
                if mode == "projection"
                else "Cornell-Berkeley Experiment 0, arbitrary SU2 tomography"
            ),
            "experiment_order": (
                "Per gate: input +Z, +X, +Y; phase-shifted prep; one IQ gate pulse; fixed final +X pi/2; dual-rail reference"
                if mode == "virtual_z"
                else "Per gate: input +Z, +X, +Y; fixed final +X pi/2; dual-rail reference"
                if mode == "projection"
                else "Per gate: Z, X, Y, dual-rail reference"
            ),
            "ref_even_rep": "ms0",
            "ref_odd_rep": "ms1",
            "acquisition_layout": (
                "counts[gate, exp(inputZ,inputX,inputY,ref), nv, run, step, rep]"
                if mode in ("projection", "virtual_z")
                else "counts[gate, exp(ZXYref), nv, run, step, rep]"
            ),
        }

        # Analyze before saving. save_raw_data replaces raw['counts']
        # with an NPZ path in the dictionary passed to it.
        result = arb_analysis.analyze(
            raw,
            min_ref_contrast=2.5 if mode in ("projection", "virtual_z")
                             else arb_analysis.MIN_REFERENCE_CONTRAST,
            min_ref_snr=3.0 if mode in ("projection", "virtual_z")
                        else arb_analysis.MIN_REFERENCE_SNR,
        )

        print()
        print(
            "Reference valid fraction:",
            result["reference_valid_fraction"],
        )

        repr_nv = widefield.get_repr_nv_sig(nv_list)
        file_path = dm.get_file_path(
            __file__,
            raw["timestamp"],
            repr_nv.name,
        )

        # IMPORTANT: use a shallow copy so raw['counts'] stays numeric.
        dm.save_raw_data(
            raw.copy(),
            file_path,
        )

        print("Saved raw data:", file_path)

        if save_plots or show_plots:
            from matplotlib import pyplot as plt

            fig, fig2 = arb_analysis.plot_results(
                raw, result, nv_ind=None,
            )
            if save_plots:
                dm.save_figure(fig, file_path)
                fig_path2 = dm.get_file_path(
                    __file__, raw["timestamp"], repr_nv.name + "-heatmap",
                )
                dm.save_figure(fig2, fig_path2)

            # Never block a calling loop. In unattended/batched mode keep only
            # the saved files, and close the figures to avoid accumulating GUI
            # windows/memory across repeated experiments.
            if show_plots:
                plt.show(block=False)
            else:
                plt.close(fig)
                plt.close(fig2)

        return raw

    except Exception:
        print(
            "Cornell arbitrary-gate tomography failed:\n"
            + traceback.format_exc()
        )
        raise

    finally:
        tb.reset_cfm()
