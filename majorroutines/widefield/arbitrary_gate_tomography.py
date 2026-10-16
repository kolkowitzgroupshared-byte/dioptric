"""Purcell multi-NV Cornell/McMahon Experiment 0.

Follows bootstrapped_pulse_error_tomography.py, base_routine.main(),
widefield.get_base_scc_seq_args(), stream_load(), dm.save_raw_data().
Does not manipulate or overwrite existing OPX configuration files.
"""
from pathlib import Path
import traceback

from majorroutines.widefield import base_routine
from utils import common, widefield
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import tool_belt as tb
from utils.single_qubit_gate_compiler import (compile_gates,save_manifest,load_manifest,
                                    verify_manifest)
from analysis import arbitrary_gate_tomography as arb_analysis

SEQ_FILE = "arbitrary_gate_tomography.py"


def _paths():
    root=Path(__file__).resolve().parents[2]
    folder=root/"calibrations"/"arbitrary_gate_tomography"
    return folder/"arbSQR.csv",folder


def prepare(uwave_ind=0):
    """Precompute JSON manifest for current chain's pi calibration (NO hardware)."""
    csv_path,folder=_paths()
    if not csv_path.exists():
        raise FileNotFoundError(f"Place Cornell CSV at {csv_path}")
    pi_ns=tb.get_virtual_sig_gen_dict(uwave_ind)["pi_pulse"]
    manifest=compile_gates(csv_path, pi_ns, uwave_ind)
    out_path=folder/f"purcell_uwave{uwave_ind}.json"
    save_manifest(manifest,out_path)
    return manifest,out_path


def main(nv_list, num_reps=8, num_runs=5, uwave_ind_list=(0,),
         allow_large_error=False, show_plots=True):
    """Run 10 gates x {Z,X,Y} with one dual-rail reference per rep.

    WARNING: running energizes the microwave, lasers and camera via base_routine.
    Run only when Purcell hardware is available and no other routine is running.
    """
    if not nv_list:
        raise ValueError("Empty NV list")
    if len(uwave_ind_list)!=1:
        raise ValueError("One MW chain at a time: use (0,), (1,), or (2,)")
    if num_reps<4 or num_reps%2:
        raise ValueError("Use even num_reps >=4 for alternating dual-rail ref")
    chain=int(uwave_ind_list[0])
    manifest,path=prepare(chain)  # recalculate from current pi calibration
    cfg=common.get_config_module()
    pi_ns=tb.get_virtual_sig_gen_dict(chain)["pi_pulse"]
    verify_manifest(manifest,cfg.opx_config,chain,pi_ns,
                    allow_large_error=allow_large_error)
    for gate in manifest["gates"]:
        print(f"Gate {gate['gate']:2d} {gate['decomposition']} : "
              f"{[(p['phase_deg'],p['duration_ns']) for p in gate['pulses']]} ns "
              f"SO3 err={gate['rotation_error_rad']:.4f}rad")

    pulse_gen=tb.get_server_pulse_gen()
    n_exps=3*len(manifest["gates"])+1

    def run_fn(shuffled_step_inds):
        seq_args=[widefield.get_base_scc_seq_args(nv_list,list(uwave_ind_list)),
                  manifest["gates"]]
        pulse_gen.stream_load(SEQ_FILE,tb.encode_seq_args(seq_args),num_reps)

    raw=base_routine.main(
        nv_list, num_steps=1,num_reps=num_reps,num_runs=num_runs,
        run_fn=run_fn,uwave_ind_list=list(uwave_ind_list),
        num_exps=n_exps, load_iq=True,
    )
    raw |= {"timestamp":dm.get_time_stamp(),
            "cornell_manifest":manifest,
            "cornell_manifest_path":str(path),
            "experiment":"Cornell-Berkeley Experiment 0, arbitrary SU2 tomography",
            "experiment_order":"Gate1 Z,X,Y; ...; Gate10 Z,X,Y; dual-rail ref",
            "ref_even_rep":"ms0", "ref_odd_rep":"ms1"}
    repr_nv=widefield.get_repr_nv_sig(nv_list)
    file_path=dm.get_file_path(__file__,raw["timestamp"],repr_nv.name)
    dm.save_raw_data(raw,file_path)
    print("Saved raw data:",file_path)
    try:
        result=arb_analysis.analyze(raw)
        print("Reference valid fraction",result["reference_valid_fraction"])
        if show_plots:
            fig,fig2=arb_analysis.plot_results(raw,result,nv_ind=0)
            dm.save_figure(fig,file_path)
            fig_path2=dm.get_file_path(__file__,raw["timestamp"],repr_nv.name+"-heatmap")
            dm.save_figure(fig2,fig_path2)
            kpl.show(block=True)
    except Exception:
        print("Raw data saved, but analysis failed:\n"+traceback.format_exc())
        raise
    finally:
        tb.reset_cfm()
    return raw
