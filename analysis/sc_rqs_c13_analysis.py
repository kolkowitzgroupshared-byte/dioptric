# -*- coding: utf-8 -*-
"""Multi-file 13C randomized-timing DD (RQS-inspired) + XY8-8 analysis.

Paste this into: C:/Users/saroj/Github/dioptric/analysis/sc_rqs_c13_analysis.py
Run from the repository root:

    python -m analysis.sc_rqs_c13_analysis --files STEM1 STEM2 STEM3
    python -m analysis.sc_rqs_c13_analysis --files STEM1 STEM2 --outdir D:/rqs_check
    python -m analysis.sc_rqs_c13_analysis --files "D:/data/NAME1.txt" "D:/data/NAME2.txt" --outdir D:/rqs_check
    python -m analysis.sc_rqs_c13_analysis --self-test

OR edit FILE_STEMS and run: python -m analysis.sc_rqs_c13_analysis

Sources are normalized **independently** against their own alternating SCC
reference. NVs are aligned by stable NV name (not index). By default, identical pulse schedules (random masks and XY8 settings) are
matched across files and their independently SCC-normalized data are averaged
into ONE trace per NV. Different masks cannot be averaged; --no-average keeps
them as distinct observations for the original multi-filter analysis.

Expected raw data: counts[exp=2, nv, run, step, rep] and rqs_c13_protocol.

OUTPUTS default to data_manager (dm) on its configured NAS/G: backend.
--outdir is an explicit local override. No silent fallback to C:.

This is a diagnostic randomized-DD filter-overlap regression, NOT Kannan's
full optimal RQS protocol, a calibrated nuclear PSD, or proof of 13C identity.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------- USER SETTINGS ----------------------------
FILE_STEMS = [
    "2026_10_22-12_21_29-johnson-nv0_2026_10_02",
    "2026_10_22-10_21_40-johnson-nv0_2026_10_02",
    
]
FILE_STEMS = [
    # "2026_10_23-14_26_03-johnson-nv0_2026_10_02",
    "2026_10_27-22_11_45-johnson-nv0_2026_10_02"
]

DATA_DIR = None                # Folder containing source .txt/.npz files, if needed
OUTDIR = None                  # None -> dm / configured NAS (normally G:)
NV_MODE = "intersection"      # "intersection" (default) or "union"
TARGET_KHZ = None             # Low: bare 13C; high: broadband exploratory (optional candidate)
FIELD_TOLERANCE_G = 0.3       # Refuse merging physically different fields
MIN_REF_CONTRAST = 0.2        # Minimum per-NV bright/dark separation (photons)
SHOW_PLOTS = False
EXPORT_PDF = True              # Save every summary plot as vector PDF
PER_NV_PDFS = True              # One detailed PDF per NV, plus multipage PDF
AVERAGE_MATCHED_FILES = True     # Average identical pulse patterns across files
QC_REFERENCE_CONTRAST = 2.5    # Photon separation used for QUALITY FLAGS only
QC_REFERENCE_SNR = 3.0         # Reference SNR flag (not a hard filter)
# --------------------------------------------------------------------------


def _stem(source):
    p = Path(str(source))
    return p.stem if p.suffix.lower() in (".txt", ".npz") else p.name


def _nv_name(nv):
    name = nv.get("name") if isinstance(nv, dict) else getattr(nv, "name", None)
    if not name:
        raise ValueError("An NV is missing a stable name; cannot align NVs across files")
    return str(name)


def _load_file(source, data_dir=None):
    """Load through dm, or read an explicit pair of .txt+.npz source files.

    Direct mode is useful on NVOffice, where a basic Python environment may
    lack labrad but still have numpy/matplotlib. It does not modify source.
    """
    path = Path(str(source))
    stem = _stem(source)
    direct = path.suffix.lower() in (".txt", ".npz") or data_dir is not None
    if direct:
        if data_dir is not None and not path.is_absolute():
            path = Path(data_dir) / path.name
        txt = path.with_suffix(".txt")
        npz = path.with_suffix(".npz")
        if not txt.is_file():
            raise FileNotFoundError(f"Missing RQS metadata: {txt}")
        if not npz.is_file():
            raise FileNotFoundError(f"Missing RQS counts: {npz}")
        raw = json.loads(txt.read_text(encoding="utf-8"))
        with np.load(npz, allow_pickle=False) as z:
            if "counts" not in z:
                raise ValueError(f"No counts array in {npz}")
            raw["counts"] = z["counts"].copy()
        return stem, raw
    try:
        from utils import data_manager as dm
    except Exception as exc:
        raise RuntimeError(
            "Cannot import lab data_manager in this Python environment. "
            "Use the usual dioptric conda environment, or supply --data-dir "
            "containing the raw .txt and .npz files."
        ) from exc
    raw = dm.get_raw_data(file_stem=stem, load_npz=True)
    return stem, raw


def _filter_power(step, freqs_hz):
    """Ideal toggling-filter |integral y(t) exp(i 2*pi*f*t)dt|^2, units s^2."""
    f = np.asarray(freqs_hz, float)
    ts = np.r_[0.0, np.asarray(step["pi_centers_ns"], float)*1e-9,
               float(step["T_ns"])*1e-9]
    signs = (-1.0)**np.arange(len(ts)-1)
    w = 2*np.pi*f
    y = np.sum((np.exp(1j*w[:, None]*ts[1:]) -
                np.exp(1j*w[:, None]*ts[:-1]))*signs, axis=1)
    nz = np.abs(w) > 1e-14
    out = np.zeros(len(f), dtype=complex)
    out[nz] = y[nz]/(1j*w[nz])
    out[~nz] = np.sum(np.diff(ts)*signs)
    return np.abs(out)**2


def _band_overlap(step, f_hz, fractional_width=0.2):
    half = max(float(f_hz)*fractional_width, 5e3)
    lo = max(float(f_hz)-half, 1e3)
    hi = float(f_hz)+half
    freq = np.linspace(lo, hi, 101)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(_filter_power(step, freq), freq)/(hi-lo))


def _high_band_overlap(step, lower_hz, upper_hz):
    """Average ideal |Y(f)|^2 over the *entire* measured high-frequency band.

    Deliberately not a resonance-selective 13C estimator. With no user-supplied
    hyperfine frequency, this gives exploratory filter-coverage diagnostics
    without inventing a nuclear candidate.
    """
    if not (0 < lower_hz < upper_hz):
        raise ValueError("High-band limits must be positive and increasing")
    freq=np.linspace(lower_hz, upper_hz, 601)
    integrate=np.trapezoid if hasattr(np,"trapezoid") else np.trapz
    return float(integrate(_filter_power(step,freq),freq)/(upper_hz-lower_hz))


def _overlap_xlabel(result):
    if result.get("overlap_mode")=="full_high_band":
        lo,hi=result["high_band_bounds_hz"]
        return f"Broadband filter overlap {lo/1e3:.0f}-{hi/1e3:.0f} kHz (1e-9 s^2)"
    return "Candidate-band |Y|^2 overlap (1e-9 s^2)"


def _has_candidate(result):
    return np.isfinite(result["target_hz"])


def _check_and_normalize(stem, raw, min_contrast=MIN_REF_CONTRAST):
    """Keep one contrast per original file, NV, and step.

    The base_routine already stores counts indexed by logical step (even when
    acquisition order step_ind_master_list is shuffled). Nothing is reordered.
    """
    proto = raw.get("rqs_c13_protocol")
    if not isinstance(proto, dict) or not proto.get("steps"):
        raise ValueError(f"{stem}: missing rqs_c13_protocol/steps")
    if raw.get("rqs_complete") is not True:
        raise ValueError(f"{stem}: rqs_complete is not True; refusing partial acquisition")
    counts = np.asarray(raw["counts"], dtype=float)
    if counts.ndim != 5 or counts.shape[0] != 2:
        raise ValueError(f"{stem}: expected counts[2,nv,run,step,rep]; got {counts.shape}")
    _, n_nv, n_runs, n_steps, n_reps = counts.shape
    if n_reps < 2 or n_reps % 2:
        raise ValueError(f"{stem}: dual-rail requires an EVEN rep count >=2")
    if len(proto["steps"]) != n_steps:
        raise ValueError(f"{stem}: protocol step count {len(proto['steps'])} != {n_steps}")
    if n_runs != int(raw.get("num_runs", n_runs)) or n_reps != int(raw.get("num_reps", n_reps)):
        raise ValueError(f"{stem}: count dimensions disagree with recorded acquisition")
    orders = raw.get("step_ind_master_list", None)
    if orders is not None and (len(orders) != n_runs or any(
        x is None or sorted(x) != list(range(n_steps)) for x in orders
    )):
        raise ValueError(f"{stem}: missing or invalid completed step order")
    nvs = raw.get("nv_list", [])
    names = [_nv_name(nv) for nv in nvs]
    if len(names) != n_nv or len(set(names)) != len(names):
        raise ValueError(f"{stem}: missing/duplicate NV names or mismatched NV count")
    if not np.isfinite(counts).all():
        raise ValueError(f"{stem}: nonfinite raw counts; check incomplete steps")

    # Each source has its own even/odd dual-rail reference. Never merge raw
    # bright/dark levels before computing the source's normalized coherence.
    signal = counts[0].mean(axis=(1, 3))       # [NV, step]
    ref_even = counts[1, :, :, :, ::2].mean(axis=(1, 3))
    ref_odd = counts[1, :, :, :, 1::2].mean(axis=(1, 3))
    delta = ref_even-ref_odd
    # Precision of dual-rail normalization (1000 even and 1000 odd frames
    # for a 20-run x 100-rep acquisition). Use raw frame scatter; do not
    # estimate the SNR from already-averaged bright/dark levels.
    ref = counts[1]
    even = ref[:, :, :, ::2]
    odd = ref[:, :, :, 1::2]
    n_each = n_runs * (n_reps // 2)
    if n_each > 1:
        sem_delta = np.sqrt((np.var(even, axis=(1,3), ddof=1) +
                             np.var(odd, axis=(1,3), ddof=1)) / n_each)
    else:
        sem_delta = np.full_like(delta, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ref_snr = np.where(sem_delta > 0, np.abs(delta)/sem_delta, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_raw = np.where(np.abs(delta) >= min_contrast,
                         (2*signal-ref_even-ref_odd)/delta, np.nan)

    # Match existing single-file script's measured-sign convention while
    # reporting how sign was chosen for every NV / every source.
    random_inds = [i for i,s in enumerate(proto["steps"]) if s["kind"] == "random"]
    if not random_inds:
        raise ValueError(f"{stem}: no randomized steps")
    short = sorted(random_inds, key=lambda i: proto["steps"][i]["T_ns"])
    short = short[:max(4, min(15, len(short)))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        orient = np.sign(np.nanmedian(c_raw[:, short], axis=1))
    orient[~np.isfinite(orient) | (orient == 0)] = 1.0
    coherence = orient[:, None]*c_raw

    return {
        "stem": stem, "proto": proto, "names": names,
        "index": {name: j for j,name in enumerate(names)},
        "coherence": coherence, "reference_delta": delta,
        "reference_snr": ref_snr, "reference_sem": sem_delta,
        "orient": orient, "shots_per_step": n_runs*n_reps,
        "num_runs": n_runs, "num_reps": n_reps,
        "microwave_chains": tuple(raw.get("uwave_ind", raw.get("uwave_ind_list", []))),
    }


def _check_compatibility(records, field_tolerance_g):
    ref = records[0]["proto"]
    for r in records[1:]:
        if r["microwave_chains"] != records[0]["microwave_chains"]:
            raise ValueError(f"{r['stem']}: microwave chains/orientations differ; analyze separately")
        p = r["proto"]
        for key in ("band", "number_of_pi_pulses", "pi_ns", "pi2_ns", "microwave_phases_deg"):
            if p.get(key) != ref.get(key):
                raise ValueError(f"{r['stem']}: protocol {key} differs; analyze separately")
        b1, b0 = float(p["field_g_expected"]), float(ref["field_g_expected"])
        if abs(b1-b0) > field_tolerance_g:
            raise ValueError(f"{r['stem']}: field {b1:g}G differs from first "
                             f"{b0:g}G by > {field_tolerance_g:g}G")
    if len({r["proto"].get("protocol_sha256") for r in records}) > 1:
        print("INFO: protocol hashes differ: combining measurements by each stored"
              " filter, NOT pretending all random masks are identical")


def _slope_with_file_duration_intercepts(x, y, group, weight):
    """WLS dephasing slope with one intercept per (source file, sensing T).

    x = band-filter overlap / 1e-9[s^2], y=-ln C. Each group is demeaned
    separately so drift/baseline T2 offsets are not mistaken for nuclear power.
    SE is descriptive only; within-file shot correlations are not accounted for.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    g = np.asarray(group)
    w = np.asarray(weight, float)
    valid = (np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0))
    if valid.sum() < 8:
        return np.nan, np.nan, int(valid.sum()), 0
    x, y, g, w = x[valid], y[valid], g[valid], w[valid]
    xx = np.zeros_like(x)
    yy = np.zeros_like(y)
    groups_used = 0
    for label in np.unique(g):
        m = g == label
        if m.sum() < 2:
            continue
        xx[m] = x[m]-np.average(x[m], weights=w[m])
        yy[m] = y[m]-np.average(y[m], weights=w[m])
        groups_used += 1
    den = np.dot(w, xx**2)
    if den < 1e-12 or groups_used < 2:
        return np.nan, np.nan, int(valid.sum()), groups_used
    slope = float(np.dot(w, xx*yy)/den)
    resid = yy-slope*xx
    df = max(1, int(valid.sum())-groups_used-1)
    se = float(np.sqrt(np.dot(w,resid**2)/df/den))
    return slope, se, int(valid.sum()), groups_used


def _analyze_records(records, target_khz, nv_mode):
    nameset = set(records[0]["names"])
    if nv_mode == "intersection":
        for rec in records[1:]:
            nameset &= set(rec["names"])
    elif nv_mode == "union":
        nameset = set().union(*(set(rec["names"]) for rec in records))
    else:
        raise ValueError("nv_mode must be intersection or union")
    # Preserve first file's order, append extra names in union mode.
    names = [n for n in records[0]["names"] if n in nameset]
    names += sorted(nameset.difference(names))
    if not names:
        raise ValueError("No common NV names across sources. Try --nv-mode union")
    high_auto=(records[0]["proto"].get("band")=="high" and target_khz is None)
    high_bounds=None
    if high_auto:
        xy=[float(st["target_f_hz"]) for st in records[0]["proto"]["steps"]
            if st["kind"]=="xy8" and st.get("target_f_hz") is not None]
        if len(xy)<2:
            raise ValueError("High bank lacks enough XY8 frequencies for full-band overlap")
        high_bounds=(min(xy),max(xy))
        f_target=float("nan")  # no candidate invented
        print("AUTO HIGH-BAND exploratory overlap: %.1f to %.1f kHz; "
              "no 13C hyperfine frequency selected."%
              (high_bounds[0]/1e3, high_bounds[1]/1e3))
    else:
        f_target=(float(target_khz)*1e3 if target_khz is not None else
                  float(records[0]["proto"]["bare_c13_larmor_hz_expected"]))
        if f_target <= 0 or not np.isfinite(f_target):
            raise ValueError("target frequency must be positive and finite")
    # Every observation retains its source and unique sensing filter.
    observations = []
    for file_index, rec in enumerate(records):
        overlaps = (
            [_high_band_overlap(st,*high_bounds) for st in rec["proto"]["steps"]]
            if high_auto else
            [_band_overlap(st,f_target) for st in rec["proto"]["steps"]]
        )
        for step_index, (s, overlap) in enumerate(zip(rec["proto"]["steps"], overlaps)):
            observations.append({
                "file_index": file_index, "step_index": step_index,
                "file_stem": rec["stem"], "kind": s["kind"],
                "T_ns": float(s["T_ns"]),
                "xy8_frequency_hz": (float(s["target_f_hz"]) if s.get("target_f_hz") is not None else np.nan),
                "filter_overlap_s2": overlap,
                "shots": rec["shots_per_step"],
                "step_signature": json.dumps([s["kind"], s.get("waits_cc"), s.get("phase_deg")], separators=(",", ":")),
            })
    n, m = len(names), len(observations)
    C = np.full((n,m), np.nan)
    ref_delta = np.full((n,m), np.nan)
    ref_snr = np.full((n,m), np.nan)
    for i, name in enumerate(names):
        for j, o in enumerate(observations):
            r = records[o["file_index"]]
            if name in r["index"]:
                k = r["index"][name]
                C[i,j] = r["coherence"][k,o["step_index"]]
                ref_delta[i,j] = r["reference_delta"][k,o["step_index"]]
                ref_snr[i,j] = r["reference_snr"][k,o["step_index"]]
    is_random = np.asarray([o["kind"] == "random" for o in observations])
    frequency_hz = np.asarray([o["xy8_frequency_hz"] for o in observations])
    foverlap = np.asarray([o["filter_overlap_s2"] for o in observations])
    t_ns = np.asarray([o["T_ns"] for o in observations])
    wt = np.asarray([o["shots"] for o in observations], float)
    group = np.asarray([f"{o['file_index']}:{int(round(o['T_ns']))}" for o in observations])
    rows = []
    slopes = np.full(n, np.nan)
    slope_se = np.full(n, np.nan)
    for i, name in enumerate(names):
        good = is_random & np.isfinite(C[i]) & (C[i] > 0.025) & (C[i] < 1.2)
        x = foverlap[good]/1e-9
        y = -np.log(np.clip(C[i,good], 0.025, 0.999))
        sl,se,num,ng = _slope_with_file_duration_intercepts(x,y,group[good],wt[good])
        slopes[i], slope_se[i] = sl,se
        rows.append({
            "nv_name": name, "nv_index_combined": i,
            "files_present": sum(name in r["index"] for r in records),
            "valid_random_points": num, "file_duration_groups": ng,
            "slope_arbitrary": sl, "slope_sem_descriptive": se,
            "median_reference_sep_photons": float(np.nanmedian(np.abs(ref_delta[i])))
            if np.isfinite(ref_delta[i]).any() else np.nan,
            "median_reference_snr": float(np.nanmedian(ref_snr[i]))
            if np.isfinite(ref_snr[i]).any() else np.nan,
            "fraction_unphysical_abs_coherence_gt_1": float(np.mean(np.abs(C[i,np.isfinite(C[i])]) > 1))
            if np.isfinite(C[i]).any() else np.nan,
            "median_random_coherence": float(np.nanmedian(C[i,is_random]))
            if np.isfinite(C[i,is_random]).any() else np.nan,
        })
    return {
        "names": names, "observations": observations, "coherence": C,
        "reference_delta": ref_delta, "reference_snr": ref_snr,
        "slopes": slopes, "slope_se": slope_se,
        "per_nv": rows, "frequency_hz": frequency_hz, "filter_overlap_s2": foverlap,
        "time_ns": t_ns, "random_mask": is_random, "shots": wt,
        "target_hz": f_target,
        "overlap_mode": "full_high_band" if high_auto else "candidate_around_frequency",
        "high_band_bounds_hz": high_bounds,
    }


# -*- coding: utf-8 -*-
"""Average matched RQS acquisitions after independent SCC normalization."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings


def average_matching(records, result, regression):
    if len(records)<2:
        result["averaged_files"]=len(records)
        return result
    steps0=records[0]["proto"]["steps"]
    for rec in records[1:]:
        if len(rec["proto"]["steps"])!=len(steps0):
            raise ValueError("Cannot average: different step count.")
        for j,(a,b) in enumerate(zip(steps0,rec["proto"]["steps"])):
            for key in ("kind","waits_cc","phase_deg","pi_centers_ns",
                        "T_ns","target_f_hz"):
                if a.get(key)!=b.get(key):
                    raise ValueError(
                        "Cannot average nonidentical masks: %s step %d %s. "
                        "Use --no-average if protocols differ."
                        %(rec["stem"],j,key))
    nf=len(records); nstep=len(steps0); n=len(result["names"])
    src=result["coherence"].reshape(n,nf,nstep)
    ref=result["reference_delta"].reshape(n,nf,nstep)
    snr=result["reference_snr"].reshape(n,nf,nstep)
    shotcount=np.array([r["shots_per_step"] for r in records],float)
    weights=np.broadcast_to(shotcount[None,:,None],src.shape)
    valid=np.isfinite(src)
    w=np.where(valid,weights,0.)
    nshot=w.sum(axis=1)
    avg=np.divide(np.where(valid,src*w,0.).sum(axis=1),nshot,
                  out=np.full((n,nstep),np.nan),where=nshot>0)
    vref=valid & np.isfinite(ref)
    wr=np.where(vref,weights,0.)
    d=wr.sum(axis=1)
    delta=np.divide(np.where(vref,ref*wr,0.).sum(axis=1),d,
                    out=np.full_like(avg,np.nan),where=d>0)
    with np.errstate(divide="ignore",invalid="ignore"):
        sem=np.abs(ref)/snr
    vs=vref & np.isfinite(sem) & (snr>0)
    ws=np.where(vs,weights,0.)
    den=ws.sum(axis=1)
    serr=np.divide(np.sqrt(np.where(vs,(weights*sem)**2,0.).sum(axis=1)),
                   den,out=np.full_like(avg,np.nan),where=den>0)
    with np.errstate(divide="ignore",invalid="ignore"):
        snr_avg=np.where(serr>0,np.abs(delta)/serr,np.nan)
    obs=[]
    for j in range(nstep):
        o=dict(result["observations"][j])
        o.update(file_index=-1,file_stem="AVERAGE_OF_%d_FILES"%nf,
                 source_count=nf,shots=int(shotcount.sum()))
        obs.append(o)
    is_rand=result["random_mask"][:nstep].copy()
    f=result["frequency_hz"][:nstep].copy()
    ov=result["filter_overlap_s2"][:nstep].copy()
    time_ns=result["time_ns"][:nstep].copy()
    group=np.round(time_ns).astype(int)
    slopes=np.full(n,np.nan); slope_se=np.full(n,np.nan); rows=[]
    for i,name in enumerate(result["names"]):
        ok=is_rand & np.isfinite(avg[i]) & (avg[i]>0.025) & (avg[i]<0.999)
        b,se,num,ng=regression(ov[ok]/1e-9,-np.log(avg[i,ok]),
                               group[ok],nshot[i,ok])
        slopes[i]=b; slope_se[i]=se
        good=np.isfinite(avg[i])
        rows.append({
            "nv_name":name,"nv_index_combined":i,
            "files_present":sum(name in r["index"] for r in records),
            "valid_random_points":num,"file_duration_groups":ng,
            "slope_arbitrary":b,"slope_sem_descriptive":se,
            "median_reference_sep_photons":(
                float(np.nanmedian(np.abs(delta[i])))
                if np.isfinite(delta[i]).any() else np.nan),
            "median_reference_snr":(
                float(np.nanmedian(snr_avg[i]))
                if np.isfinite(snr_avg[i]).any() else np.nan),
            "fraction_unphysical_abs_coherence_gt_1":(
                float(np.mean(np.abs(avg[i,good])>1)) if good.any() else np.nan),
            "median_random_coherence":(
                float(np.nanmedian(avg[i,is_rand]))
                if np.isfinite(avg[i,is_rand]).any() else np.nan),
            "min_contributing_files":int(valid[i].sum(axis=0).min()),
            "max_contributing_files":int(valid[i].sum(axis=0).max())
        })
    result.update(observations=obs,coherence=avg,reference_delta=delta,
                  reference_snr=snr_avg,slopes=slopes,slope_se=slope_se,
                  per_nv=rows,frequency_hz=f,filter_overlap_s2=ov,
                  time_ns=time_ns,random_mask=is_rand,
                  shots=np.full(nstep,shotcount.sum()),
                  effective_shots_per_nv_step=nshot,
                  contributing_files_per_nv_step=valid.sum(axis=1),
                  averaged_files=nf)
    print("AVERAGE %d files -> %d settings (%d random filters and %d XY8)"
          %(nf,nstep,is_rand.sum(),(~is_rand).sum()))
    return result


def averaged_figures(result, xy8_pooled):
    """Present ONE combined curve rather than overlays."""
    f,pooled,num=xy8_pooled(result)
    C=result["coherence"]
    rnd=result["random_mask"]
    overlap=result["filter_overlap_s2"]/1e-9
    t=result["time_ns"]/1e3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        median=np.nanmedian(C,axis=0)
    figs={}
    fig,axes=plt.subplots(3,1,figsize=(10,11),layout="constrained")
    xy=np.flatnonzero(~rnd)
    xy=xy[np.argsort(result["frequency_hz"][xy])]
    axes[0].plot(result["frequency_hz"][xy]/1e3,median[xy],"o-",
                 label="Mean of matching acquisitions")
    if _has_candidate(result):
        axes[0].axvline(result["target_hz"]/1e3,ls="--",c="black",label="target frequency")
    axes[0].set(xlabel="XY8 frequency (kHz)",ylabel="Median NV coherence",
                title="Average of %d acquisitions"%result["averaged_files"])
    axes[0].legend();axes[0].grid(alpha=.2)
    r=np.flatnonzero(rnd)
    axes[1].scatter(t[r],median[r],c=overlap[r],cmap="viridis")
    uniq=np.unique(t[r])
    means=[np.nanmean(median[r][t[r]==x]) for x in uniq]
    axes[1].plot(uniq,means,"k--",label="Average across masks")
    axes[1].set(xlabel="Sensing time (us)",ylabel="Median NV coherence",
                title="Averaged randomized experiments")
    axes[1].legend();axes[1].grid(alpha=.2)
    im=axes[2].scatter(overlap[r],median[r],c=t[r],cmap="plasma")
    fig.colorbar(im,ax=axes[2],label="Sensing time (us)")
    axes[2].set(xlabel=_overlap_xlabel(result),
                ylabel="Median NV coherence",
                title="One point per distinct random pattern")
    axes[2].grid(alpha=.2)
    figs["averaged_overview"]=fig

    fig,ax=plt.subplots(figsize=(10,max(5,.16*len(result["names"])+2)),
                        layout="constrained")
    if len(f):
        image=ax.imshow(pooled,origin="lower",aspect="auto",
                        interpolation="nearest",vmin=-1,vmax=1,cmap="viridis")
        fig.colorbar(image,ax=ax,label="Averaged coherence")
        ix=np.unique(np.linspace(0,len(f)-1,min(8,len(f))).astype(int))
        ax.set_xticks(ix,[f"{f[k]/1e3:.0f}" for k in ix])
    ax.set(xlabel="XY8 frequency (kHz)",ylabel="NV index",
           title="Per-NV averaged XY8 coherence")
    figs["averaged_xy8_heatmap"]=fig

    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    valid=np.isfinite(result["slopes"])
    i=np.flatnonzero(valid)
    yerr=np.where(np.isfinite(result["slope_se"][valid]),
                  result["slope_se"][valid],0.)
    ax.errorbar(i,result["slopes"][valid],yerr=yerr,fmt=".",capsize=2)
    ax.axhline(0,c="black",lw=.8)
    ax.set(xlabel="NV index",ylabel="Band overlap slope (arb.)",
           title=f"Averaged random-mask slope ({valid.sum()}/{len(valid)} finite)")
    ax.grid(alpha=.2)
    figs["averaged_regression_by_nv"]=fig

    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    absref=np.nanmedian(np.abs(result["reference_delta"]),axis=0)
    ax.plot(np.arange(len(absref)),absref,"o-")
    ax.set(xlabel="Pulse setting",ylabel="Reference separation (photons)",
           title="Reference quality after averaging acquisitions")
    ax.grid(alpha=.2)
    figs["averaged_reference_quality"]=fig
    return figs,(f,pooled,num)


def averaged_nv_figure(i,result,ref_snr_threshold=3):
    """Per-NV PDF: one XY8 trace and one value for each matched RQS mask."""
    C=result["coherence"][i]; rnd=result["random_mask"]
    f=result["frequency_hz"]/1e3
    t=result["time_ns"]/1e3
    overlap=result["filter_overlap_s2"]/1e-9
    fig,axs=plt.subplots(3,1,figsize=(9,10),layout="constrained")
    xy=np.flatnonzero(~rnd)
    xy=xy[np.argsort(f[xy])]
    axs[0].plot(f[xy],C[xy],"o-",label="Mean of matching files")
    if _has_candidate(result):
        axs[0].axvline(result["target_hz"]/1e3,c="black",ls="--",
                       label="target frequency")
    axs[0].set(xlabel="XY8 frequency (kHz)",ylabel="NV coherence",
               title="Averaged XY8: ONE curve")
    axs[0].set_ylim(-1.3,1.3)
    axs[0].legend(); axs[0].grid(alpha=.2)

    r=np.flatnonzero(rnd)
    image=axs[1].scatter(overlap[r],C[r],c=t[r],cmap="plasma")
    fig.colorbar(image,ax=axs[1],label="T (us)")
    axs[1].set(xlabel=_overlap_xlabel(result),
               ylabel="NV coherence",
               title="Averaged random mask measurements")
    axs[1].set_ylim(-1.3,1.3);axs[1].grid(alpha=.2)

    rsnr=result["reference_snr"][i]
    axs[2].plot(np.arange(len(rsnr)),rsnr,"o-")
    axs[2].axhline(ref_snr_threshold,c="black",ls="--",label="QC threshold")
    axs[2].set(xlabel="Pulse setting",ylabel="Reference SNR",
               title="Pooled SCC reference quality")
    axs[2].legend();axs[2].grid(alpha=.2)
    slope=result["slopes"][i];err=result["slope_se"][i]
    ref=result["reference_delta"][i]
    med=np.nanmedian(np.abs(ref)) if np.isfinite(ref).any() else np.nan
    bad=int(np.sum(np.abs(C[np.isfinite(C)])>1))
    name=result["names"][i]
    fig.suptitle(f"{name}: averaged {result['averaged_files']} files\n"
                 f"beta={slope:.3g} +/- {err:.3g} (descriptive); "
                 f"|ref|={med:.2f} photons; {bad} unphysical values",fontsize=10)
    return fig


def _xy8_pooled(result):
    """Per-NV pooled normalized XY8 contrast at matching frequencies.

    Frequencies differing by <=0.5 Hz are put into a common 1-Hz bin.
    Other frequencies are NOT interpolated or treated as equivalent.
    """
    f = result["frequency_hz"]
    c = result["coherence"]
    shots = result["shots"]
    valid_freqs = np.isfinite(f) & ~result["random_mask"]
    keys = np.array([int(round(v)) for v in f[valid_freqs]], dtype=int)
    grid = sorted(set(keys.tolist()))
    arr = np.full((len(result["names"]),len(grid)), np.nan)
    n_src = np.zeros_like(arr, dtype=int)
    for k,freq in enumerate(grid):
        mask = np.zeros_like(f, bool)
        mask[np.where(valid_freqs)[0][keys==freq]] = True
        for i in range(c.shape[0]):
            ok = mask & np.isfinite(c[i])
            if np.any(ok):
                arr[i,k] = float(np.average(c[i,ok],weights=shots[ok]))
                n_src[i,k] = int(np.sum(ok))
    return np.asarray(grid,float), arr, n_src


def _make_figures(records, result):
    if result.get("averaged_files",0)>1:
        return averaged_figures(result,_xy8_pooled)
    figs = {}
    f, pooled, pooled_num = _xy8_pooled(result)
    c = result["coherence"]
    is_rand = result["random_mask"]
    ov = result["filter_overlap_s2"]/1e-9
    T = result["time_ns"]/1000
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(3,1, figsize=(11,12), layout="constrained")
    # Show every file's measured XY8 response without pretending unmatched
    # XY8 frequencies are on the same grid.
    for k, rec in enumerate(records):
        cols=[j for j,o in enumerate(result["observations"])
              if o["file_index"]==k and o["kind"]=="xy8"]
        if not cols: continue
        fr = result["frequency_hz"][cols]/1e3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(c[:,cols], axis=0)
        ix=np.argsort(fr)
        axes[0].plot(fr[ix],med[ix],".-",label=rec["stem"][:19],alpha=.75,
                     color=colors[k%len(colors)])
    if _has_candidate(result):
        axes[0].axvline(result["target_hz"]/1e3,color="black",ls="--",lw=1,label="target")
    axes[0].set(xlabel="XY8 nominal frequency (kHz)",ylabel="Median NV coherence",
                title="Independently normalized XY8 per acquisition")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    if len(f):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            med=np.nanmedian(pooled,axis=0)
        axes[1].plot(f/1e3,med,"o-",label="Shot-weighted at matching frequencies")
    if _has_candidate(result):
        axes[1].axvline(result["target_hz"]/1e3,color="black",ls="--",lw=1)
    axes[1].set(xlabel="XY8 frequency (kHz)",ylabel="Median NV coherence",
                title="Combined XY8 (no interpolation across different grids)")
    axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        med=np.nanmedian(c,axis=0)
    sc=axes[2].scatter(ov[is_rand],med[is_rand],c=T[is_rand],cmap="plasma",s=22)
    fig.colorbar(sc,ax=axes[2],label="Sensing time (us)")
    axes[2].set(xlabel=_overlap_xlabel(result),
                ylabel="Median NV coherence",
                title="Random mask response; every source/step retained")
    axes[2].grid(alpha=.2)
    figs["multifile_overview"] = fig

    fig,ax=plt.subplots(figsize=(11,max(4,0.19*len(result["names"])+2)),layout="constrained")
    if pooled.shape[1]:
        im=ax.imshow(pooled,aspect="auto",interpolation="nearest",origin="lower",cmap="viridis")
        fig.colorbar(im,ax=ax,label="SCC-normalized coherence")
        ticks=np.linspace(0,len(f)-1,min(8,len(f)),dtype=int)
        ax.set_xticks(ticks,[f"{f[v]/1e3:.0f}" for v in ticks])
    ax.set(xlabel="XY8 nominal frequency (kHz)",ylabel="Common NV index",
           title="Combined spatially resolved XY8 response")
    figs["xy8_nv_heatmap"] = fig

    fig,ax=plt.subplots(figsize=(11,4.8),layout="constrained")
    valid=np.isfinite(result["slopes"])
    inds=np.arange(len(result["names"]))
    ax.errorbar(inds[valid],result["slopes"][valid],
                yerr=np.where(np.isfinite(result["slope_se"][valid]),
                              result["slope_se"][valid],0),
                fmt="o",markersize=3,capsize=2)
    ax.axhline(0,color="black",lw=.7)
    ax.set(xlabel="Common NV index",ylabel="Band-overlap slope (arbitrary)",
           title=f"Random-mask regression by NV: {valid.sum()}/{len(valid)} finite")
    ax.grid(alpha=.2)
    figs["random_regression_by_nv"] = fig

    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    for k,rec in enumerate(records):
        vals=np.abs(rec["reference_delta"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            median=np.nanmedian(vals,axis=0)
        ax.plot(np.arange(len(median)),median,marker=".",lw=.75,label=rec["stem"][:19])
    ax.set(xlabel="Logical step index within file",ylabel="Median reference separation (photons)",
           title="Independent SCC reference quality by acquisition")
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    figs["reference_contrast_sources"] = fig
    return figs, (f,pooled,pooled_num)



def _one_nv_pdf_figure(i, records, result):
    """Three-panel diagnosis for one NV; never interpret a lone slope as 13C."""
    if result.get("averaged_files",0)>1:
        return averaged_nv_figure(i,result,ref_snr_threshold=QC_REFERENCE_SNR)
    name = result["names"][i]
    c = result["coherence"][i]
    overlap = result["filter_overlap_s2"] / 1e-9
    t_us = result["time_ns"] / 1e3
    f_hz = result["frequency_hz"]
    obs = result["observations"]
    sep = result["reference_delta"][i]
    snr = result["reference_snr"][i]
    fig, axs = plt.subplots(3, 1, figsize=(9.0, 10.4), layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, rec in enumerate(records):
        inds = [j for j, o in enumerate(obs)
                if o["file_index"] == k and o["kind"] == "xy8"]
        if inds:
            fr = f_hz[inds] / 1e3
            order = np.argsort(fr)
            axs[0].plot(fr[order], c[np.asarray(inds)[order]], "o-",
                        color=colors[k % len(colors)],
                        label=f"{rec['stem'][:19]}", ms=3, alpha=.78)
    if _has_candidate(result):
        axs[0].axvline(result["target_hz"] / 1e3, ls="--", c="black", lw=1,
                       label="target frequency")
    axs[0].axhline(0, color="grey", lw=.6)
    axs[0].set(xlabel="Nominal XY8 frequency (kHz)",
               ylabel="SCC-normalized coherence",
               title="XY8-8 response: frequencies also change sensing duration")
    axs[0].set_ylim(-1.25, 1.25)
    axs[0].grid(alpha=.2)
    axs[0].legend(fontsize=7)

    for k, rec in enumerate(records):
        inds = np.array([j for j, o in enumerate(obs)
                         if o["file_index"] == k and o["kind"] == "random"], int)
        if inds.size:
            axs[1].scatter(overlap[inds], c[inds], s=24,
                           color=colors[k % len(colors)],
                           label=rec["stem"][:19], alpha=.8)
            # Duration labels expose if apparent overlap response is just T2.
            for j in inds:
                if np.isfinite(c[j]) and -1.25 <= c[j] <= 1.25:
                    axs[1].annotate(f"{t_us[j]:.0f}", (overlap[j], c[j]),
                                    fontsize=6, alpha=.7, xytext=(2,2),
                                    textcoords="offset points")
    axs[1].set(xlabel=_overlap_xlabel(result),
               ylabel="SCC-normalized coherence",
               title="Random-mask measurements (labels = T in microseconds)")
    axs[1].set_ylim(-1.25, 1.25)
    axs[1].grid(alpha=.2)
    axs[1].legend(fontsize=7)

    for k, rec in enumerate(records):
        inds = [j for j, o in enumerate(obs) if o["file_index"] == k]
        axs[2].plot(range(len(inds)), snr[inds], ".-",
                    color=colors[k % len(colors)], label=rec["stem"][:19])
    axs[2].axhline(QC_REFERENCE_SNR, c="black", ls="--",
                   lw=1, label=f"SNR = {QC_REFERENCE_SNR:.0f} QC guide")
    axs[2].set(xlabel="Logical step within each acquisition",
               ylabel="SCC reference separation / standard error",
               title="Dual-rail SCC reference SNR (computed from raw frames)")
    axs[2].grid(alpha=.2)
    axs[2].legend(fontsize=7)

    slope, se = result["slopes"][i], result["slope_se"][i]
    medref = np.nanmedian(np.abs(sep)) if np.isfinite(sep).any() else np.nan
    medsnr = np.nanmedian(snr) if np.isfinite(snr).any() else np.nan
    nbad = int(np.sum(np.abs(c[np.isfinite(c)]) > 1.0))
    quality = ("LOW REFERENCE CONTRAST" if medref < QC_REFERENCE_CONTRAST
               else "reference separation OK")
    n_bad_snr = int(np.sum(np.isfinite(snr) & (snr < QC_REFERENCE_SNR)))
    if nbad:
        quality += f"; {nbad} unphysical coherence points"
    if n_bad_snr:
        quality += f"; {n_bad_snr} low-SNR reference steps"
    fig.suptitle(
        f"NV {i:03d}: {name}\n"
        f"band slope = {slope:.3g} +/- {se:.3g} (descriptive), "
        f"median |ref| = {medref:.2f} photons, median ref SNR = {medsnr:.2f}\n"
        f"QC: {quality}", fontsize=10,
    )
    return fig


def _pdf_bytes(fig):
    """Matplotlib vector PDF bytes (the PNG exports remain unchanged)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight")
    return buf.getvalue()


def _safe_filepart(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_"
                   for ch in str(value))[:90]



def _csv_text(rows, headers):
    mem=io.StringIO(newline="")
    wr=csv.DictWriter(mem,fieldnames=headers,extrasaction="ignore")
    wr.writeheader();wr.writerows(rows)
    return mem.getvalue()


def _save(records, result, figures, xy8_arrays, summary, outdir):
    from numpy import savez_compressed
    f, pooled, pooled_num = xy8_arrays
    nvrows = result["per_nv"]
    files=[{
        "file_stem": r["stem"],"nvs":len(r["names"]),
        "runs":r["num_runs"],"reps":r["num_reps"],
        "steps":len(r["proto"]["steps"]),"shots_per_step":r["shots_per_step"],
        "microwave_chains":str(r["microwave_chains"]),
        "fraction_NV_sign_reversed":float(np.mean(r["orient"]<0)),
        "protocol_sha256":r["proto"].get("protocol_sha256"),
        "mean_abs_reference_delta":float(np.nanmean(abs(r["reference_delta"]))),
        "fraction_valid_contrast":float(np.mean(np.isfinite(r["coherence"]))),
    } for r in records]
    obs_rows=[]
    for o in result["observations"]:
        obs_rows.append({k:v for k,v in o.items() if k != "step_signature"})
    data_arrays={
        "nv_names":np.asarray(result["names"],dtype=str),
        "source_stems":np.asarray([r["stem"] for r in records],dtype=str),
        "coherence_by_nv_observation":result["coherence"],
        "effective_shots_by_nv_step":result.get("effective_shots_per_nv_step",np.empty((0,0))),
        "contributing_files_by_nv_step":result.get("contributing_files_per_nv_step",np.empty((0,0))),
        "reference_delta_by_nv_observation":result["reference_delta"],
        "reference_snr_by_nv_observation":result["reference_snr"],
        "band_overlap_s2":result["filter_overlap_s2"],
        "sensing_time_ns":result["time_ns"],
        "nominal_xy8_frequency_hz":result["frequency_hz"],
        "is_random_mask":result["random_mask"],
        "shots_per_observation":result["shots"],
        "file_index_by_observation":np.asarray([o["file_index"] for o in result["observations"]]),
        "step_index_by_observation":np.asarray([o["step_index"] for o in result["observations"]]),
        "regression_slopes":result["slopes"],
        "regression_slope_se":result["slope_se"],
        "xy8_frequency_hz":f,"pooled_xy8_coherence":pooled,
        "xy8_measurements_per_nv_frequency":pooled_num,
    }
    csv_outputs={
        "per_nv":_csv_text(nvrows,list(nvrows[0]) if nvrows else []),
        "per_file":_csv_text(files,list(files[0]) if files else []),
        "observations":_csv_text(obs_rows,list(obs_rows[0]) if obs_rows else []),
    }
    summary_text=json.dumps(summary,indent=2,allow_nan=False)
    if outdir is None:
        try:
            from utils import data_manager as dm
            timestamp=dm.get_time_stamp()
            base=dm.get_file_path(__file__,timestamp,"rqs_c13_multifile")
            for name,fig in figures.items():
                dm.save_figure(fig,base.with_name(base.stem+"-"+name+".txt"))
            for name,content in csv_outputs.items():
                dm.save_text(content,base.with_name(base.stem+"-"+name+".txt"),suffix=".csv")
            dm.save_text(summary_text,base.with_name(base.stem+"-summary.txt"),suffix=".json")
            dm.save_npz(base,compressed=True,**data_arrays)
            print("Saved via data_manager to:",base.parent)
        except ImportError as exc:
            raise RuntimeError("data_manager import failed (e.g. labrad missing). "
                               "Use the lab Python environment or an explicit --outdir.") from exc
    else:
        folder=Path(outdir);folder.mkdir(parents=True,exist_ok=True)
        for name,fig in figures.items():
            fig.savefig(folder/(name+".png"),dpi=180,bbox_inches="tight")
        for name,content in csv_outputs.items():
            (folder/(name+".csv")).write_text(content,encoding="utf-8")
        (folder/"summary.json").write_text(summary_text,encoding="utf-8")
        savez_compressed(folder/"combined_results.npz",**data_arrays)
        print("Saved explicitly to:",folder)
    return base if outdir is None else folder



def _export_pdfs(records, result, figures, outdir, saved_path,
                 per_nv_pdfs=PER_NV_PDFS):
    """Every summary figure as vector PDF; optional 119 individual NV PDFs.

    Uses dm's configured cloud/NAS uploader for binary PDF bytes. This keeps
    PDF files on the SAME backend as PNG/CSV/NPZ, without local fallback.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if outdir is None:
        from utils import data_manager as dm
        base = Path(saved_path)

        def put_pdf(relative_name, payload):
            path = base.parent / relative_name
            dm.cloud.upload(path, payload)
            return path

        def summary_name(name):
            return f"{base.stem}-{name}.pdf"

        per_nv_folder = f"{base.stem}-individual_nv_pdfs"
        all_nv_filename = f"{base.stem}-all_nv_diagnostics.pdf"
    else:
        folder = Path(saved_path)

        def put_pdf(relative_name, payload):
            path = folder / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return path

        def summary_name(name):
            return name + ".pdf"

        per_nv_folder = "individual_nv_pdfs"
        all_nv_filename = "all_nv_diagnostics.pdf"

    for name, fig in figures.items():
        p = put_pdf(summary_name(name), _pdf_bytes(fig))
        print("Vector PDF:", p)

    if not per_nv_pdfs:
        return

    combined = io.BytesIO()
    count = 0
    with PdfPages(combined) as book:
        for i, name in enumerate(result["names"]):
            nvfig = _one_nv_pdf_figure(i, records, result)
            filename = f"nv_{i:03d}_{_safe_filepart(name)}.pdf"
            put_pdf(f"{per_nv_folder}/{filename}", _pdf_bytes(nvfig))
            book.savefig(nvfig, bbox_inches="tight")
            plt.close(nvfig)
            count += 1
            if count % 20 == 0:
                print(f"Exported {count}/{len(result['names'])} individual NV PDFs")
    all_path = put_pdf(all_nv_filename, combined.getvalue())
    print(f"Saved {count} individual NV PDFs and combined {count}-page PDF:", all_path)


def analyze_many(file_stems=FILE_STEMS, *, data_dir=DATA_DIR, outdir=OUTDIR,
                 target_khz=TARGET_KHZ,nv_mode=NV_MODE,
                 field_tolerance_g=FIELD_TOLERANCE_G,
                 min_ref_contrast=MIN_REF_CONTRAST,show_plots=SHOW_PLOTS,
                 export_pdf=EXPORT_PDF,per_nv_pdfs=PER_NV_PDFS,
                 average_matching_files=AVERAGE_MATCHED_FILES):
    """Main entry point. Does not edit raw files or manipulate hardware."""
    sources=list(file_stems)
    if not sources:
        raise ValueError("No RQS source files: edit FILE_STEMS or pass --files STEM1 STEM2 ...")
    stems=[_stem(s) for s in sources]
    if len(set(stems)) != len(stems):
        raise ValueError("The same source file was listed more than once")
    records=[]
    for source in sources:
        stem,raw=_load_file(source,data_dir=data_dir)
        records.append(_check_and_normalize(stem,raw,min_contrast=min_ref_contrast))
        print("Loaded",stem,":",len(records[-1]["names"]),"NVs,",
              records[-1]["num_runs"],"runs,",len(records[-1]["proto"]["steps"]),"steps")
    _check_compatibility(records,field_tolerance_g)
    result=_analyze_records(records,target_khz,nv_mode)
    if average_matching_files:
        result=average_matching(records,result,_slope_with_file_duration_intercepts)
    figures,xy8_arrays=_make_figures(records,result)
    valid=np.isfinite(result["slopes"])
    summary={
        "description":"RQS-inspired 13C multi-file randomized DD and periodic XY8 comparison",
        "warning":"Classical Gaussian-noise filter-overlap diagnostic; not a calibrated PSD or proof of nuclear identity",
        "file_stems":stems,"num_files":len(stems),
        "microwave_chains_per_file":[list(r["microwave_chains"]) for r in records],
        "field_g_expected_per_file":[float(r["proto"]["field_g_expected"]) for r in records],
        "target_khz":float(result["target_hz"]/1e3) if _has_candidate(result) else None,
        "overlap_mode":result["overlap_mode"],
        "high_band_bounds_khz":([v/1e3 for v in result["high_band_bounds_hz"]]
                               if result["high_band_bounds_hz"] is not None else None),
        "min_reference_contrast_threshold_used_photons":float(min_ref_contrast),
        "export_vector_pdf":bool(export_pdf),
        "export_per_nv_pdf":bool(per_nv_pdfs),
        "averaged_identical_files":bool(average_matching_files),
        "observations_after_average":len(result["observations"]),
        "bands":sorted(set(r["proto"]["band"] for r in records)),
        "nv_matching":nv_mode,"number_of_nvs":len(result["names"]),
        "number_of_random_observations":int(result["random_mask"].sum()),
        "number_of_xy8_observations":int((~result["random_mask"]).sum()),
        "nvs_with_finite_slope":int(valid.sum()),
        "median_slope_arbitrary":float(np.median(result["slopes"][valid])) if valid.any() else None,
        "positive_slope_count":int(np.sum(result["slopes"][valid] > 0)),
        "negative_slope_count":int(np.sum(result["slopes"][valid] < 0)),
        "unique_random_filter_count":len({o["step_signature"] for o in result["observations"] if o["kind"] == "random"}),
        "fraction_abs_coherence_gt_one":float(np.mean(np.abs(result["coherence"][np.isfinite(result["coherence"])]) > 1)),
        "nvs_median_reference_contrast_below_2_5":int(sum(
            np.isfinite(row).any() and
            np.nanmedian(np.abs(row)) < QC_REFERENCE_CONTRAST
            for row in result["reference_delta"])),
        "median_reference_snr":float(np.nanmedian(result["reference_snr"])),
        "qc_note":"Regression SEM descriptive, not robust significance. Inspect unphysical |C|>1 and reference SNR.",
        "combination":"Per-file SCC reference normalization; name-aligned NVs; "
                      "random masks kept as distinct observations, file-and-duration centered fit; "
                      "XY8 exact-matched frequencies shot-weighted",
    }
    try:
        saved_path=_save(records,result,figures,xy8_arrays,summary,outdir)
        if export_pdf:
            _export_pdfs(records,result,figures,outdir,saved_path,
                         per_nv_pdfs=per_nv_pdfs)
    finally:
        if show_plots:
            plt.show(block=False)
        else:
            for fig in figures.values():
                plt.close(fig)
    print("=== RQS C13 MULTI-FILE ===")
    if _has_candidate(result):
        print(f"Files: {len(records)}, NVs: {len(result['names'])}, candidate: {result['target_hz']/1e3:.3f} kHz")
    else:
        lo,hi=result["high_band_bounds_hz"]
        print(f"Files: {len(records)}, NVs: {len(result['names'])}, exploratory HIGH-band {lo/1e3:.0f}-{hi/1e3:.0f} kHz (no target)")
    print(f"Valid per-NV filter-overlap slopes: {valid.sum()}/{len(valid)}")
    if not valid.any():
        print("NOTICE: no reliable regression slopes. More INDEPENDENT random "
              "filters and reference-quality measurements are needed.")
    return {"result":result,"summary":summary,"records":records}


def self_test():
    """Create two synthetic datasets with reversed NV order and different masks."""
    from tempfile import TemporaryDirectory
    rng=np.random.default_rng(42)
    with TemporaryDirectory() as td:
        root=Path(td)
        sources=[]
        for fileno in range(2):
            names=["test_nv_A","test_nv_B"] if fileno==0 else ["test_nv_B","test_nv_A"]
            f_l=33.29e3
            steps=[]
            for dur in (64e3,96e3,128e3,160e3):
                for j in range(4):
                    T=float(dur)
                    # Sorted eight-pulse centers, different masks across files.
                    t=np.sort(rng.uniform(1000,T-1000,8))
                    steps.append({"kind":"random","T_ns":T,
                                  "pi_centers_ns":t.tolist(),"waits_cc":[1+j,fileno],
                                  "phase_deg":[0,90,0,90,90,0,90,0],"target_f_hz":None})
            for f in (25e3,33e3,50e3,90e3):
                T=8/(2*f)*1e9
                centers=(np.arange(8)+.5)*T/8
                steps.append({"kind":"xy8","T_ns":T,
                              "pi_centers_ns":centers.tolist(),"waits_cc":[1],
                              "phase_deg":[0,90,0,90,90,0,90,0],"target_f_hz":f})
            proto={"steps":steps,"band":"low","field_g_expected":31.1,
                   "bare_c13_larmor_hz_expected":f_l,"number_of_pi_pulses":8,
                   "pi_ns":96,"pi2_ns":48,
                   "microwave_phases_deg":[0,90,0,90,90,0,90,0],
                   "protocol_sha256":"synthetic"+str(fileno)}
            runs,reps=3,4
            counts=np.zeros((2,2,runs,len(steps),reps),float)
            o=np.array([_band_overlap(s,f_l)/1e-9 for s in steps])
            for ni,name in enumerate(names):
                true_c=np.exp(-.04-(.025 if name=="test_nv_A" else .055)*o)
                for rr in range(runs):
                    for j,ci in enumerate(true_c):
                        counts[0,ni,rr,j,:]=150+40*ci+rng.normal(0,1,reps)
                        counts[1,ni,rr,j,::2]=190+rng.normal(0,1,reps//2)
                        counts[1,ni,rr,j,1::2]=110+rng.normal(0,1,reps//2)
            raw={"rqs_complete":True,"rqs_c13_protocol":proto,"nv_list":[{"name":n} for n in names],
                 "num_runs":runs,"num_reps":reps,
                 "step_ind_master_list":[list(range(len(steps))) for _ in range(runs)]}
            stem=f"synthetic_test_{fileno}"
            (root/(stem+".txt")).write_text(json.dumps(raw),encoding="utf-8")
            np.savez_compressed(root/(stem+".npz"),counts=counts)
            sources.append(stem)
        out=root/"results"
        # Different masks are deliberately kept separate for this test.
        res=analyze_many(sources,data_dir=root,outdir=out,
                         average_matching_files=False)
        assert res["summary"]["num_files"]==2
        assert len(res["result"]["names"])==2
        assert np.isfinite(res["result"]["slopes"]).any()
        assert (out/"multifile_overview.png").is_file()
        assert (out/"combined_results.npz").is_file()
        assert (out/"per_nv.csv").is_file()
        print("PASS multi-file, reversed NV order, different random masks and outputs")
        # Matched masks must collapse exactly to one average per setting.
        import shutil
        copy_stem = "synthetic_same_masks_copy"
        for ext in (".txt", ".npz"):
            shutil.copyfile(root/(sources[0]+ext),root/(copy_stem+ext))
        averaged = analyze_many([sources[0],copy_stem], data_dir=root,
                                outdir=root/"averaged_results",
                                export_pdf=False, average_matching_files=True)
        a=averaged["result"]
        assert len(a["observations"]) == len(averaged["records"][0]["proto"]["steps"])
        assert a["coherence"].shape[1] == len(a["observations"])
        assert np.allclose(a["coherence"],averaged["records"][0]["coherence"],
                           equal_nan=True)
        assert a["averaged_files"] == 2
        print("PASS identical filters collapse to one correctly averaged point per setting")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files",nargs="+",default=None,
                        help="Source file stems OR .txt paths; overrides FILE_STEMS")
    parser.add_argument("--data-dir",default=DATA_DIR,
                        help="Explicit folder with the paired raw .txt+.npz files")
    parser.add_argument("--outdir",default=OUTDIR,
                        help="Explicit output folder. Default saves using dm to G:/NAS")
    parser.add_argument("--nv-mode",choices=("intersection","union"),default=NV_MODE)
    parser.add_argument("--target-khz",type=float,default=TARGET_KHZ)
    parser.add_argument("--field-tolerance-g",type=float,default=FIELD_TOLERANCE_G)
    parser.add_argument("--min-ref-contrast",type=float,default=MIN_REF_CONTRAST)
    parser.add_argument("--show",action="store_true",help="Nonblocking figure display")
    parser.add_argument("--no-pdf",action="store_true",help="Disable vector PDF exports")
    parser.add_argument("--no-average",action="store_true",help="Show individual file curves (default: average identical masks)")
    parser.add_argument("--no-per-nv-pdfs",action="store_true",help="Save summary figure PDFs only")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    analyze_many(FILE_STEMS if args.files is None else args.files,
                 data_dir=args.data_dir,outdir=args.outdir,target_khz=args.target_khz,
                 nv_mode=args.nv_mode,field_tolerance_g=args.field_tolerance_g,
                 min_ref_contrast=args.min_ref_contrast,show_plots=args.show,
                 export_pdf=not args.no_pdf, per_nv_pdfs=not args.no_per_nv_pdfs,
                 average_matching_files=not args.no_average)


if __name__=="__main__":
    main()
