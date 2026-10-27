"""RQS fixed-duration multisine acquisition adapter."""
import numpy as np
from utils.rqs_multisine_core import compile_bank,choose_masks
from majorroutines.widefield import base_routine
from utils import tool_belt as tb,data_manager as dm,widefield

FREQUENCIES_HZ=tuple(np.linspace(234375,1415625,8))

def construct(duration_us=8.,n_settings=96,bank_size=8192,
              seed=37,pi_ns=96,pi2_ns=48,
              coupling_hz=2800.,max_pulses=48):
    bank=compile_bank(FREQUENCIES_HZ,duration_us=duration_us,
                      coupling_hz=coupling_hz,seed=seed,
                      pool_size=bank_size,pi_ns=pi_ns,
                      max_pulses=max_pulses)
    rows=choose_masks(bank,n_settings=n_settings,
                      x_fraction=.2,seed=seed+2)
    largest=max(x["n_pi"] for x in rows)
    full_cc=round(bank.duration_s/4e-9)
    pi_cc=round(pi_ns/4)
    for r in rows:
        ts=r["switch_ticks"]
        if ts:
            gaps=[int(ts[0]-pi_cc//2)]
            gaps += [int(b-a-pi_cc) for a,b in zip(ts[:-1],ts[1:])]
            gaps += [int(full_cc-ts[-1]-pi_cc+pi_cc//2)]
        else:
            gaps=[int(full_cc)]
        if min(gaps)<4 or sum(gaps)+pi_cc*len(ts)!=full_cc:
            raise ValueError("Pulse timing does not fit hardware")
        r["gaps_cc"]=gaps+[4]*(largest+1-len(gaps))
    return bank,rows


def main(nv_list,*,duration_us=8.,n_settings=96,bank_size=8192,
         seed=37,coupling_hz=2800.,num_reps=20,num_runs=20,
         uwave_ind_list=(1,),phase_reference_confirmed=False):
    if not phase_reference_confirmed:
        raise ValueError(
            "Phase-locked source required at EVERY sensing shot. "
            "Confirm external phase reset/trigger before acquiring.")
    if len(uwave_ind_list)!=1 or not nv_list:
        raise ValueError("Specify calibrated NVs and exactly one MW chain")
    if num_reps<2 or num_reps%2 or num_runs<1:
        raise ValueError("SCC dual-rail needs positive runs and even reps")
    uw=int(uwave_ind_list[0])
    cfg=tb.get_virtual_sig_gen_dict(uw)
    pi_ns=int(cfg["pi_pulse"])
    pi2_ns=int(cfg["pi_on_2_pulse"])
    bank,rows=construct(duration_us,n_settings,bank_size,
                        seed,pi_ns,pi2_ns,coupling_hz)
    print("RQS fixed time:",duration_us,"us,",
          n_settings,"settings,",num_runs,"runs,",
          num_reps,"reps, covariance condition",bank.condition)
    base_args=widefield.get_base_scc_seq_args(nv_list,[uw])
    pulse_gen=tb.get_server_pulse_gen()
    max_used=max(x["n_pi"] for x in rows)
    def run_fn(order):
        seq=[rows[int(i)] for i in order]
        waits=[[int(r["gaps_cc"][k]) for r in seq]
               for k in range(max_used+1)]
        count=[int(r["n_pi"]) for r in seq]
        bases=[0 if r["basis"]=="X" else 1 for r in seq]
        parity=[1 if r["n_pi"]%2==0 else 0 for r in seq]
        pulse_gen.stream_load(
            "rqs_multisine.py",
            tb.encode_seq_args([base_args,waits,count,bases,parity]),
            num_reps)
    raw=base_routine.main(
        nv_list=nv_list,num_steps=len(rows),
        num_reps=num_reps,num_runs=num_runs,
        run_fn=run_fn,uwave_ind_list=[uw],num_exps=2,
        ref_by_rep_parity=True,load_iq=True)
    complete=sum(x is not None for x in raw["step_ind_master_list"])
    proto={"experiment":"rqs_fixed_T_multisine",
           "field_mode":"phase_locked",
           "fixed_T_us":duration_us,
           "frequencies_hz":bank.frequencies_hz.tolist(),
           "coupling_hz":coupling_hz,
           "bank_size":2*len(bank.responses),
           "seed":seed,"n_settings":n_settings,
           "pi_ns":pi_ns,"pi2_ns":pi2_ns,
           "clock_s":bank.clock_s,
           "bank_condition":bank.condition,
           "rows":rows,
           "bank_suffix":"-bank.npz",
           "readout_sign_calibration_required":True}
    raw.update(timestamp=dm.get_time_stamp(),
               rqs_multisine_protocol=proto,
               rqs_complete=bool(complete==num_runs))
    path=dm.get_file_path(
        __file__,raw["timestamp"],
        widefield.get_repr_nv_sig(nv_list).name)
    maxp=bank.max_pulses
    centers=np.asarray(
        [list(x)+[-1]*(maxp-len(x)) for x in bank.pulses_ticks],
        dtype=np.int32)
    dm.save_npz(
        path.with_name(path.stem+"-bank"),compressed=True,
        frequencies_hz=bank.frequencies_hz,
        pulse_center_ticks=centers,initial_sign=bank.initial_sign,
        response_vectors=bank.responses,covariance=bank.sigma,
        duration_s=np.asarray(bank.duration_s),
        clock_s=np.asarray(bank.clock_s),
        coupling_hz=np.asarray(bank.coupling_hz),
        condition=np.asarray(bank.condition))
    dm.save_raw_data(raw,path)
    print("RQS data saved:",path)
    if complete!=num_runs:
        raise RuntimeError("Incomplete acquisition saved and flagged")
    tb.reset_cfm()
    return raw

if __name__ == '__main__':
    bank, rows = construct(bank_size=128, n_settings=24)
    print('Offline compiler test', len(rows), bank.condition)
