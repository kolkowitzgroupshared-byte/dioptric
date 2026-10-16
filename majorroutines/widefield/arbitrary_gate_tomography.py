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

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.single_qubit_gate_compiler import (
    compile_gates,
    save_manifest,
    verify_manifest,
)
from analysis import arbitrary_gate_tomography as arb_analysis


SEQ_FILE = "arbitrary_gate_tomography.py"


def _paths():
    root = Path(__file__).resolve().parents[2]
    folder = root / "calibrations" / "purcell" / "arbitrary_gate_tomography"
    return folder / "arbSQR.csv", folder


def prepare(uwave_ind=0):
    """Precompute manifest using the current chain's pi calibration."""
    csv_path, folder = _paths()

    if not csv_path.exists():
        raise FileNotFoundError(f"Place Cornell CSV at {csv_path}")

    pi_ns = tb.get_virtual_sig_gen_dict(uwave_ind)["pi_pulse"]
    manifest = compile_gates(csv_path, pi_ns, uwave_ind)

    out_path = folder / f"purcell_uwave{uwave_ind}.json"
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
    show_plots=True,
):
    """Run Cornell Experiment 0 on many NVs.

    Each of the 10 arbitrary gates is compiled and acquired separately.
    Per gate, the OPX program contains only:
        Z readout, X readout, Y readout, dual-rail reference.

    Final counts shape:
        [gate, experiment=4, NV, run, step=1, repetition]
    """

    if not nv_list:
        raise ValueError("Empty NV list")

    if len(uwave_ind_list) != 1:
        raise ValueError("One MW chain at a time: use (0,), (1,), or (2,)")

    if num_reps < 4 or num_reps % 2:
        raise ValueError("Use even num_reps >= 4 for alternating dual-rail ref")

    chain = int(uwave_ind_list[0])

    manifest, manifest_path = prepare(chain)

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
    print("Cornell arbitrary-gate manifest")
    print("--------------------------------")
    for gate in manifest["gates"]:
        print(
            f"Gate {gate['gate']:2d} {gate['decomposition']} : "
            f"{[(p['phase_deg'], p['duration_ns']) for p in gate['pulses']]} ns "
            f"SO3 err={gate['rotation_error_rad']:.4f} rad"
        )

    pulse_gen = tb.get_server_pulse_gen()

    gate_raw_list = []

    try:
        for gate_ind, gate in enumerate(manifest["gates"]):
            print()
            print("=" * 70)
            print(
                f"Cornell gate {gate_ind + 1}/{len(manifest['gates'])}: "
                f"{gate['decomposition']}"
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

                # One gate -> Z,X,Y + automatically appended reference.
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
        # Combine all 10 independently acquired gate blocks.
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
                "Cornell-Berkeley Experiment 0, arbitrary SU2 tomography"
            ),
            "experiment_order": (
                "Per gate: Z, X, Y, dual-rail reference"
            ),
            "ref_even_rep": "ms0",
            "ref_odd_rep": "ms1",
            "acquisition_layout": (
                "counts[gate, exp(ZXYref), nv, run, step, rep]"
            ),
        }

        # Analyze before saving. save_raw_data replaces raw['counts']
        # with an NPZ path in the dictionary passed to it.
        result = arb_analysis.analyze(raw)

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

        if show_plots:
            fig, fig2 = arb_analysis.plot_results(
                raw,
                result,
                nv_ind=0,
            )

            dm.save_figure(
                fig,
                file_path,
            )

            fig_path2 = dm.get_file_path(
                __file__,
                raw["timestamp"],
                repr_nv.name + "-heatmap",
            )

            dm.save_figure(
                fig2,
                fig_path2,
            )

            kpl.show(block=True)

        return raw

    except Exception:
        print(
            "Cornell arbitrary-gate tomography failed:\n"
            + traceback.format_exc()
        )
        raise

    finally:
        tb.reset_cfm()
