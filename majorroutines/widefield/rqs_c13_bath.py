# -*- coding: utf-8 -*-
"""Random-multisine pi-pulse sensing of an unpolarized 13C spin bath.

This is NOT the phase-locked classical RQS Fourier-amplitude experiment.
Only SCC-normalized X coherence is measured; Y quadrature, signed masks,
and the linear phase estimator are intentionally omitted.

Variable-count pi_X sensing masks, fixed T within a dataset. Includes
periodic 8-pulse XY8 and spin echo as same-T controls. Each mask is drawn
without replacement from the random-multisine bank; the original bank's
4 ns physical pulse positions are retained. No source phase synchronization
is assumed. Do not call main until device calibration is complete.
"""
from __future__ import annotations
import json
import hashlib
import numpy as np
from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import tool_belt as tb
from utils import widefield
from utils.rqs_multisine_core import compile_bank

GAMMA_C13_HZ_PER_G = 1070.5
GRID_BY_T = {
    40: (8_000., 33_292.55, 58_585.1),
    60: (15_000., 33_292.55, 51_600.),
    80: (20_000., 33_292.55, 46_600.),
}


def _edges(frequencies_hz):
    f=np.asarray(sorted(frequencies_hz),float)
    if len(f)<2:
        width=max(1e3,float(f[0])*.25)
        return [float(max(1.,f[0]-width)),float(f[0]+width)]
    middle=(f[1:]+f[:-1])*.5
    first=max(1.,float(f[0]-(middle[0]-f[0])))
    return [first]+middle.tolist()+[float(f[-1]+f[-1]-middle[-1])]


def build_protocol(*, field_g=31.1, duration_us=60.,
                   frequencies_hz=None, seed=37, n_masks=64,
                   bank_size=8192, pi_ns=96, pi2_ns=48,
                   max_pulses=48, coupling_hz=2800.):
    if not np.isfinite(field_g) or field_g<=0:
        raise ValueError("field_g must be the expected measured field")
    if frequencies_hz is None:
        rounded=int(round(duration_us))
        if abs(duration_us-rounded)>.01 or rounded not in GRID_BY_T:
            raise ValueError("Supply frequencies_hz or choose T=40,60,80 us")
        frequencies_hz=GRID_BY_T[rounded]
    freq=np.asarray(frequencies_hz,dtype=float)
    if len(freq)<1 or np.any(freq<=0) or np.unique(freq).size!=len(freq):
        raise ValueError("Frequencies must be unique and positive")
    if not 1<=n_masks<=bank_size//2:
        raise ValueError("n_masks must be <= half bank_size")
    if not 0<=pi2_ns and pi_ns%4==0 and pi2_ns%4==0:
        raise ValueError("Microwave pulse lengths must be 4ns multiples")
    bank=compile_bank(freq,duration_us=duration_us,coupling_hz=coupling_hz,
         seed=seed,pool_size=bank_size,clock_ns=4.,pi_ns=pi_ns,
         buffer_ns=16.,max_pulses=max_pulses)
    rng=np.random.default_rng(seed+117)
    inds=rng.choice(len(bank.pulses_ticks),n_masks,replace=False)
    total_cc=round(bank.duration_s/4e-9)
    pi_cc=pi_ns//4
    rows=[]
    def add_row(centers,kind,bank_index=None,phase_mode=0):
        centers=[int(x) for x in centers]
        n=len(centers)
        if centers!=sorted(centers):
            raise ValueError("Pulse centers must be sorted")
        if n:
            gaps=[int(centers[0]-pi_cc//2)]
            gaps.extend(int(b-a-pi_cc) for a,b in zip(centers[:-1],centers[1:]))
            gaps.append(int(total_cc-centers[-1]-pi_cc+pi_cc//2))
        else:
            gaps=[int(total_cc)]
        if min(gaps)<4 or sum(gaps)+n*pi_cc!=total_cc:
            raise ValueError("Physical pi-pulse overlap or incorrect total T")
        rows.append({
           "kind":kind,"bank_index":None if bank_index is None else int(bank_index),
           "n_pi":n,"switch_ticks":centers,
           "pi_centers_ns":[int(x*4) for x in centers],
           "T_ns":int(4*total_cc),
           "gaps_cc":gaps,
           "phase_mode":int(phase_mode),
           "readout_basis":"X",
        })
    for k in inds:
        add_row(bank.pulses_ticks[int(k)],"random_multisine",int(k))
    # Control 1: periodic XY8; same T and 8 pi centers.
    xy8=np.rint((np.arange(8)+.5)*total_cc/8).astype(int)
    add_row(xy8,"periodic_xy8",phase_mode=1)
    # At T=60 us, XY4 targets approximately 33.3 kHz; XY8 targets 66.7 kHz.
    xy4=np.rint((np.arange(4)+.5)*total_cc/4).astype(int)
    add_row(xy4,"periodic_xy4",phase_mode=2)
    # Control 3: Hahn echo at same T; one pi center.
    add_row([total_cc//2],"spin_echo",phase_mode=0)
    maximum=max(r["n_pi"] for r in rows)
    for r in rows:
        r["gaps_cc_padded"]=r.pop("gaps_cc")+[4]*(maximum+1-(r["n_pi"]+1))
        assert len(r["gaps_cc_padded"])==maximum+1
    protocol={
        "experiment":"rqs_c13_bath_fixed_T",
        "measurement":"SCC_normalized_X_coherence",
        "model":"random_multisine_sign_filters_conditional_quantum_bath",
        "field_g_expected":float(field_g),
        "c13_larmor_hz_expected":float(field_g*GAMMA_C13_HZ_PER_G),
        "T_us":float(duration_us),"num_masks":int(n_masks),
        "frequency_centers_hz":freq.tolist(),
        "frequency_edges_hz":_edges(freq),
        "bank_size":int(bank_size),"bank_seed":int(seed),
        "bank_covariance_condition":float(bank.condition),
        "pi_ns":int(pi_ns),"pi2_ns":int(pi2_ns),
        "max_pi":int(maximum),"max_pulses_cap":int(max_pulses),
        "clock_ns":4,
        "notes":"Unpolarized 13C bath: no AC phase-lock assumption. "
                "Classical PSD screening is approximate; verify with quantum model.",
        "steps":rows,
    }
    protocol["protocol_sha256"]=hashlib.sha256(
        json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return protocol


def main(nv_list, *, field_g=31.1,duration_us=60.,
         frequencies_hz=None,seed=37,n_masks=64,bank_size=8192,
         uwave_ind_list=(1,),num_reps=50,num_runs=40):
    """Acquire RQS bath data; intended to be called from Purcell control panel."""
    if not nv_list or len(uwave_ind_list)!=1:
        raise ValueError("Provide nonempty calibrated NV list and single MW chain")
    if num_reps<2 or num_reps%2 or num_runs<1:
        raise ValueError("Require even num_reps and positive num_runs")
    uw=int(uwave_ind_list[0])
    cfg=tb.get_virtual_sig_gen_dict(uw)
    proto=build_protocol(field_g=field_g,duration_us=duration_us,
         frequencies_hz=frequencies_hz,seed=seed,n_masks=n_masks,
         bank_size=bank_size,pi_ns=int(cfg["pi_pulse"]),
         pi2_ns=int(cfg["pi_on_2_pulse"]))
    rows=proto["steps"]
    maxpi=proto["max_pi"]
    base_args=widefield.get_base_scc_seq_args(nv_list,[uw])
    pulse_gen=tb.get_server_pulse_gen()
    print("13C bath RQS:",len(rows),"settings, fixed T=",duration_us,
          "us; pi pulse range",min(r["n_pi"] for r in rows),maxpi)
    def run_fn(shuffled_step_inds):
        chosen=[rows[int(i)] for i in shuffled_step_inds]
        gaps=[[int(s["gaps_cc_padded"][k]) for s in chosen]
              for k in range(maxpi+1)]
        n_pi=[int(s["n_pi"]) for s in chosen]
        phase=[int(s["phase_mode"]) for s in chosen]
        even=[1 if s["n_pi"]%2==0 else 0 for s in chosen]
        pulse_gen.stream_load("rqs_c13_bath.py",
            tb.encode_seq_args([base_args,gaps,n_pi,phase,even]),num_reps)
    raw=base_routine.main(
        nv_list=nv_list,num_steps=len(rows),num_reps=num_reps,
        num_runs=num_runs,run_fn=run_fn,
        uwave_ind_list=[uw],num_exps=2,
        ref_by_rep_parity=True,load_iq=True)
    complete=sum(x is not None for x in raw["step_ind_master_list"])
    raw.update(timestamp=dm.get_time_stamp(),
        rqs_c13_bath_protocol=proto,
        rqs_complete=(complete==num_runs),
        rqs_completed_runs=int(complete),
        rqs_expected_runs=int(num_runs))
    path=dm.get_file_path(__file__,raw["timestamp"],
         widefield.get_repr_nv_sig(nv_list).name)
    dm.save_raw_data(raw,path)
    print("Saved random-multisine 13C bath raw:",path)
    if complete!=num_runs:
        raise RuntimeError("Partial run saved and explicitly marked incomplete")
    tb.reset_cfm()
    return raw


if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--dry-run",action="store_true",required=True)
    a.add_argument("--T-us",type=float,default=60.)
    a.add_argument("--n-masks",type=int,default=16)
    a.add_argument("--bank-size",type=int,default=128)
    x=a.parse_args()
    p=build_protocol(duration_us=x.T_us,n_masks=x.n_masks,
                     bank_size=x.bank_size)
    print("Hardware-free RQS 13C bath test:",len(p["steps"]),
        "filters, covariance condition",p["bank_covariance_condition"],
        "pulse counts",[s["n_pi"] for s in p["steps"]])
