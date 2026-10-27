# -*- coding: utf-8 -*-
"""True all-X randomized-pi sensing pilot with XY8 comparison.

The RANDOM arm is eight calibrated pi_X pulses at randomized times.
The XY8 arm uses the same historical periodic XY8-8 baseline. Both use
the existing polarize -> microwave -> SCC -> camera and dual-rail reference.

readout='cos': final pi/2_X, for nuclear coherence / PSD regression.
readout='both': also final pi/2_Y for signed weak-field phase tomography.
For stochastic unpolarized 13C, phase quadrature averages to zero; use 'cos'.

This is a hardware-feasible RQS-inspired protocol, NOT a reproduction of the
optimal randomized-quantum-spectroscopy theorem in Kannan's slides.
"""
from __future__ import annotations
import copy
import hashlib
import json

from majorroutines.widefield import base_routine
from majorroutines.widefield import rqs_c13 as original_rqs
from utils import data_manager as dm
from utils import tool_belt as tb
from utils import widefield


def build_protocol(*, field_g=31.1, band="high", seed=37,
                   masks_per_duration=16, pi_ns=96, pi2_ns=48,
                   readout="cos"):
    if readout not in ("cos", "both"):
        raise ValueError("readout must be 'cos' or 'both'")
    src = original_rqs.build_protocol(
        field_g=field_g, band=band, seed=seed,
        masks_per_duration=masks_per_duration,
        pi_ns=pi_ns, pi2_ns=pi2_ns,
    )
    steps=[]
    for mask_i, step in enumerate(src["steps"]):
        if step["kind"] == "random":
            phases=(0,)*8
            bases=(0,90) if readout=="both" else (0,)
            axis=1
            kind="random_x"
        else:
            phases=(0,90,0,90,90,0,90,0)
            bases=(0,)  # keep periodic XY8 baseline, single readout quadrature
            axis=0
            kind="periodic_xy8"
        for final_phase in bases:
            entry=copy.deepcopy(step)
            entry.update(
                kind=kind, pulse_axis_mode=int(axis),
                phase_deg=list(phases),
                final_pi2_phase_deg=int(final_phase),
                mask_id=int(mask_i),
                is_phase_sensitive=bool(final_phase==90),
            )
            steps.append(entry)
    proto={k:v for k,v in src.items() if k!="steps"}
    proto.update(
        experiment="rqs_c13_random_pi_x_with_periodic_xy8_reference",
        method="random pi_X timing with optional phase quadrature; NOT Kannan theorem",
        readout=readout,
        steps=steps,
        random_x_phases_deg=[0]*8,
        comparison_xy8_phases_deg=[0,90,0,90,90,0,90,0],
    )
    base=json.dumps(proto,sort_keys=True,separators=(",",":"))
    proto["protocol_sha256"]=hashlib.sha256(base.encode()).hexdigest()
    return proto


def main(nv_list, *, field_g=31.1, band="high", uwave_ind_list=(1,),
         seed=37, masks_per_duration=16, num_reps=100, num_runs=20,
         readout="cos"):
    """Perform acquisition; call explicitly from the live Purcell session.

    No physical acquisition is run by importing this file. Check the generated
    pulse timing and microwave calibration on a small subset before full run.
    """
    if not nv_list:
        raise ValueError("Select nonempty calibrated NV list")
    inds=list(uwave_ind_list)
    if len(inds)!=1 or inds[0] not in (0,1,2):
        raise ValueError("Use exactly one microwave channel/orientation")
    if num_reps < 2 or num_reps % 2 or num_runs < 1:
        raise ValueError("Use positive runs and EVEN num_reps >= 2")
    cfg=tb.get_virtual_sig_gen_dict(inds[0])
    proto=build_protocol(
        field_g=field_g, band=band, seed=seed,
        masks_per_duration=masks_per_duration,
        pi_ns=int(cfg["pi_pulse"]),
        pi2_ns=int(cfg["pi_on_2_pulse"]),
        readout=readout,
    )
    print("RQS random_pi_X (%s), %s, %d steps x %d reps x %d runs" %
          (readout, band, len(proto["steps"]), num_reps, num_runs))
    print("Effective rep shots/step = %d, 2 SCC images per rep" %
          (num_reps*num_runs))
    seq_args_base=widefield.get_base_scc_seq_args(nv_list,inds)
    pulse_gen=tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        chosen=[proto["steps"][i] for i in shuffled_step_inds]
        waits=[[s["waits_cc"][j] for s in chosen] for j in range(9)]
        axes=[s["pulse_axis_mode"] for s in chosen]
        phases=[s["final_pi2_phase_deg"] for s in chosen]
        seq_args=[seq_args_base, waits, axes, phases]
        pulse_gen.stream_load(
            "rqs_c13_random_x.py",
            tb.encode_seq_args(seq_args),
            num_reps,
        )

    raw=base_routine.main(
        nv_list=nv_list, num_steps=len(proto["steps"]),
        num_reps=num_reps, num_runs=num_runs, run_fn=run_fn,
        uwave_ind_list=inds, num_exps=2,
        ref_by_rep_parity=True, load_iq=True,
    )
    completed=sum(x is not None for x in raw["step_ind_master_list"])
    raw.update(
        timestamp=dm.get_time_stamp(),
        rqs_c13_protocol=proto,
        rqs_complete_runs=int(completed),
        rqs_expected_runs=int(num_runs),
        rqs_complete=bool(completed==num_runs),
    )
    path=dm.get_file_path(
        __file__,raw["timestamp"],widefield.get_repr_nv_sig(nv_list).name
    )
    dm.save_raw_data(raw,path)
    print("Saved RQS pi_X RAW:",path)
    if not raw["rqs_complete"]:
        raise RuntimeError("Partial acquisition saved; explicitly marked incomplete")
    tb.reset_cfm()
    return raw


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true",required=True)
    p.add_argument("--field-g",type=float,default=31.1)
    p.add_argument("--band",choices=["low","high"],default="high")
    p.add_argument("--seed",type=int,default=37)
    p.add_argument("--masks",type=int,default=16)
    p.add_argument("--readout",choices=["cos","both"],default="cos")
    p.add_argument("--pi-ns",type=int,default=96)
    p.add_argument("--pi2-ns",type=int,default=48)
    args=p.parse_args()
    from collections import Counter
    proto=build_protocol(
        field_g=args.field_g,band=args.band,seed=args.seed,
        masks_per_duration=args.masks,pi_ns=args.pi_ns,
        pi2_ns=args.pi2_ns,readout=args.readout,
    )
    assert all(len(s["waits_cc"])==9 and min(s["waits_cc"])>=8
               for s in proto["steps"])
    assert all(abs(s["T_ns"]-s["requested_T_ns"])<=4 for s in proto["steps"])
    print("SAFE dry run (no hardware)")
    print("settings",len(proto["steps"]),
          "types",dict(Counter(s["kind"] for s in proto["steps"])),
          "pulse durations",args.pi_ns,args.pi2_ns,
          "SHA",proto["protocol_sha256"])
