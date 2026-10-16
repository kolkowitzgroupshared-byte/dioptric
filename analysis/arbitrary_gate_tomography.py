"""Postprocess raw data from Cornell Experiment 0; non-destructive.

Counts from base_routine.main:
    [experiment, NV, run, step=1, repetition]
Experiments: (gate 1 Z,X,Y), ..., (gate 10 Z,X,Y), (dual-rail ref).
Reference for even repetition: |0>.  Odd repetition: |1>.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt



def analyze(raw_data, min_ref_contrast=0.5):
    counts = np.asarray(raw_data["counts"], dtype=float)
    gates = raw_data["cornell_manifest"]["gates"]
    n_gates = len(gates)
    n_nvs = counts.shape[1]
    if counts.ndim != 5 or counts.shape[0] != 3*n_gates+1 or counts.shape[3] != 1:
        raise ValueError("Unexpected counts shape (need 3*gates+1, nv, run, 1, rep)")
    if counts.shape[-1] < 2:
        raise ValueError("Reference requires both even and odd repetitions")

    data = counts[:, :, :, 0, :]
    ref = data[-1]                          # [nv,run,rep]
    ref0 = np.nanmean(ref[:,:,0::2],axis=-1)  # [nv,run]
    ref1 = np.nanmean(ref[:,:,1::2],axis=-1)
    contrast = ref0-ref1
    valid = np.isfinite(contrast) & (np.abs(contrast)>=min_ref_contrast)
    # Avoid division by zero, do not hide negative contrast; save signs.
    denominator = np.where(valid,contrast,np.nan)
    sig = np.nanmean(data[:-1],axis=-1)   # [gate*basis,nv,run]
    sz = (2*sig-ref0[None,:,:]-ref1[None,:,:])/denominator[None,:,:]
    # per-run normalized observable -> gate,nv,run,basis
    per_run = sz.reshape(n_gates,3,n_nvs,counts.shape[2]).transpose(0,2,3,1)
    # seq basis order Z X Y -> store XYZ
    per_run = per_run[:,:,:,[1,2,0]]
    with np.errstate(invalid="ignore"):
        measured = np.nanmean(per_run,axis=2)  # gate,nv,xyz
    if counts.shape[2] >= 2:
        with np.errstate(invalid="ignore",divide="ignore"):
            n_valid = np.sum(np.isfinite(per_run),axis=2)
            ste = np.nanstd(per_run,axis=2,ddof=1)/np.sqrt(n_valid)
    else:
        ste = np.full_like(measured,np.nan)
    expected = np.array([g["expected_bloch_xyz"] for g in gates])
    compiled_expected = np.array([g["predicted_actual_bloch"] for g in gates])
    residual = measured-expected[:,None,:]
    residual_compiled = measured-compiled_expected[:,None,:]
    norm = np.linalg.norm(residual,axis=-1)
    return {"measured_xyz": measured, "ste_xyz": ste,
            "expected_xyz": expected, "compiled_xyz": compiled_expected,
            "residual_xyz": residual, "residual_compiled_xyz": residual_compiled,
            "bloch_error_norm":norm,
            "reference_0":ref0,"reference_1":ref1,
            "reference_contrast": contrast, "valid_ref":valid,
            "reference_valid_fraction":float(np.mean(valid)),
            "per_run_xyz":per_run}


def plot_results(raw_data, analysis, nv_ind=0):
    exp = analysis["expected_xyz"]
    obs = analysis["measured_xyz"]
    err = analysis["ste_xyz"]
    if nv_ind >= obs.shape[1]:
        raise ValueError("nv_ind out of range")
    n = len(exp)
    x = np.arange(1,n+1)
    fig,axs=plt.subplots(3,1,figsize=(9,9),sharex=True)
    for j,(name,ax) in enumerate(zip(("X","Y","Z"),axs)):
        ax.plot(x,exp[:,j],"k--",marker="o",label="Cornell predicted")
        ax.plot(x,analysis["compiled_xyz"][:,j],".",color="tab:orange",label="OPX quantized expected")
        ax.errorbar(x,obs[:,nv_ind,j],yerr=err[:,nv_ind,j] if np.any(np.isfinite(err[:,nv_ind,j])) else None,
                    fmt="o",color="tab:blue",label="Measured")
        ax.set_ylabel(f"<sigma_{name.lower()}>")
        ax.set_ylim(-1.2,1.2)
        ax.grid(alpha=0.2)
    axs[0].legend(fontsize=8)
    axs[-1].set_xlabel("Cornell gate index")
    fig.suptitle(f"Cornell Experiment 0, NV index {nv_ind}")
    fig.tight_layout()
    fig2,ax2=plt.subplots(figsize=(9,4))
    im=ax2.imshow(analysis["bloch_error_norm"].T,aspect="auto",origin="lower",vmin=0)
    ax2.set_xlabel("Gate index (0 based)")
    ax2.set_ylabel("NV index")
    ax2.set_title("Full Bloch-vector distance to Cornell prediction")
    fig2.colorbar(im,ax=ax2,label="|r_measured - r_target|")
    fig2.tight_layout()
    return fig,fig2


def analyze_file(file_id, nv_ind=0):
    from utils import data_manager as dm
    raw=dm.get_raw_data(file_id=file_id)
    results=analyze(raw)
    fig,fig2=plot_results(raw,results,nv_ind=nv_ind)
    print("Reference valid fraction:",results["reference_valid_fraction"])
    print("Mean |Bloch error| for each gate:",np.nanmean(results["bloch_error_norm"],axis=1))
    plt.show(block=True)
    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--file-id",required=True)
    p.add_argument("--nv-ind",type=int,default=0)
    args=p.parse_args()
    analyze_file(args.file_id,args.nv_ind)
