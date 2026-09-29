# -*- coding: utf-8 -*-
"""V39 target-guided empirical ODMR uncertainty for Johnson multifield data."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

REPO = Path(r"C:\Users\saroj\Github\dioptric")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import analysis.sc_resonance_analysis_optimized as ra
from utils import widefield

D_GHZ = 2.8785
GAMMA_MHZ_G = 2.8025
MAX_WINDOW_GHZ = 0.012
N_BOOT = 2500
N_MC = 30000
RNG = np.random.default_rng(39001)
AXES = np.array([[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1]], float) / np.sqrt(3)

OUT = Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo"
           r"\c13_spin_echo_v39_target_guided_bfield_uncertainty\2026_09")
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = {
    "49G": dict(
        B=np.array([-46.27557688,-17.16599864,-5.70139829], float),
        primary="2025_10_23-08_33_06-johnson-nv0_2025_10_21",
        companion="2025_11_09-10_40_49-johnson-nv0_2025_10_21",
        note="312-NV all-orientation ESR establishes four branches; later 204-NV scan checks two populated branches. V32 later uses catalog-rescaled 49G vector.",
    ),
    "59G": dict(
        B=np.array([-41.57848995,-32.77145194,-27.5799348], float),
        primary="2025_11_28-01_53_35-johnson-nv0_2025_10_21",
        companion="2025_11_29-04_02_02-johnson-nv0_2025_10_21",
        note="312-NV ESR primary; 204-NV ESR companion.",
    ),
    "62G": dict(
        B=np.array([-48.67047318,-32.07615947,22.49657427], float),
        primary="2025_12_10-10_28_25-johnson-nv0_2025_10_21",
        companion="2025_12_20-06_01_33-johnson-nv0_2025_10_21",
        note="312-NV ESR primary; 204-NV ESR companion.",
    ),
    "65G": dict(
        B=np.array([-31.61263115,-56.58135644,-6.5512002], float),
        primary="2025_11_20-09_14_44-johnson-nv0_2025_10_21",
        companion="2025_11_21-06_06_26-johnson-nv0_2025_10_21",
        note="312-NV ESR primary; 204-NV ESR companion.",
    ),
}
def targets_from_B(B):
    proj = AXES @ np.asarray(B, float)
    targets = D_GHZ - GAMMA_MHZ_G * np.abs(proj) / 1000.0
    signs = np.sign(proj).astype(int)
    return targets, signs

def solve_from_axis_freqs(freqs_axis, signs):
    f = np.asarray(freqs_axis, float)
    p_abs = (D_GHZ - f) * 1000.0 / GAMMA_MHZ_G
    proj = np.asarray(signs, float) * p_abs
    B, *_ = np.linalg.lstsq(AXES, proj, rcond=None)
    return B

def angle_deg(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    c=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c,-1,1))))

def robust_summary(vals):
    vals=np.asarray(vals,float)
    med=float(np.median(vals))
    mad=float(1.4826*np.median(np.abs(vals-med)))
    boots=np.median(RNG.choice(vals,(N_BOOT,len(vals)),replace=True),axis=1)
    return med, float(np.std(boots,ddof=1)), mad, len(vals)
def fit_scan(field, role, stem, B_ref):
    print(f"\n=== {field} {role}: {stem} ===")
    _, nv_list, freqs, sig, ref = ra.load_and_combine([stem])
    avg, ste = widefield.process_counts(nv_list, sig, ref, threshold=True)
    avg=np.asarray(avg,float); ste=np.asarray(ste,float)
    freqs=np.asarray(freqs,float)
    dense=np.linspace(np.nanmin(freqs),np.nanmax(freqs),45)
    with parallel_backend("loky", inner_max_num_threads=1):
        fits=Parallel(n_jobs=-1,verbose=3)(
            delayed(ra.fit_one_nv)(i,freqs,avg,ste,dense)
            for i in range(len(nv_list))
        )
    targets, signs = targets_from_B(B_ref)
    sep=np.abs(targets[:,None]-targets[None,:])
    sep[sep==0]=np.inf
    windows=np.minimum(MAX_WINDOW_GHZ,0.45*np.min(sep,axis=1))
    records=[]
    for i,r in enumerate(fits):
        if not r["success"]:
            continue
        amp1,amp2,f1,f2,width,bg = map(float,r["popt"])
        if not (np.isfinite(f1) and np.isfinite(f2) and f1 < D_GHZ < f2):
            continue
        j=int(np.argmin(np.abs(targets-f1)))
        dist=float(abs(f1-targets[j]))
        records.append((i,f1,f2,width,float(r["red_chi2"]),j,dist))
    rdf=pd.DataFrame(records,columns=[
        "nv","f1_GHz","f2_GHz","width_native","red_chi2","axis_idx","dist_GHz"
    ])
    rdf["window_GHz"]=windows[rdf["axis_idx"].to_numpy(int)]
    rdf["assigned"]=rdf["dist_GHz"] <= rdf["window_GHz"]
    rdf.to_csv(OUT/f"{field}_{role}_per_nv_fits.csv",index=False)

    rows=[]
    medians=np.full(4,np.nan); ses=np.full(4,np.nan); mads=np.full(4,np.nan)
    ns=np.zeros(4,int)
    for j in range(4):
        vals=rdf.loc[rdf["assigned"] & (rdf["axis_idx"]==j),"f1_GHz"].to_numpy(float)
        if len(vals) < 2:
            continue
        med,se,mad,n=robust_summary(vals)
        medians[j]=med; ses[j]=se; mads[j]=mad; ns[j]=n
        rows.append(dict(field=field,role=role,axis_idx=j,target_GHz=targets[j],
                         window_MHz=windows[j]*1000,
                         median_GHz=med,delta_target_MHz=(med-targets[j])*1000,
                         bootstrap_se_MHz=se*1000,robust_spread_MHz=mad*1000,n=n,
                         scan_step_MHz=float(np.median(np.diff(freqs))*1000)))
    print("targets:",np.round(targets,7))
    print("medians:",np.round(medians,7))
    print("n:",ns)
    return rows, medians, ses, mads, ns, signs
def main():
    all_rows=[]
    brows=[]
    details={}
    pinv=np.linalg.pinv(AXES)
    for field,cfg in FIELDS.items():
        rp,mp,sep,madp,npop,signs=fit_scan(field,"primary",cfg["primary"],cfg["B"])
        rc,mc,sec,madc,nc,signs2=fit_scan(field,"companion",cfg["companion"],cfg["B"])
        all_rows.extend(rp); all_rows.extend(rc)
        if not np.array_equal(signs,signs2):
            raise RuntimeError(f"{field}: sign mismatch")

        targets,_=targets_from_B(cfg["B"])
        primary_pop_se=np.divide(
            madp,np.sqrt(npop),out=np.full(4,np.nan),where=npop>0
        )
        companion_pop_se=np.divide(
            madc,np.sqrt(nc),out=np.full(4,np.nan),where=nc>0
        )
        stat=np.nanmax(np.vstack([sep,sec,primary_pop_se,companion_pop_se]),axis=0)
        repro_obs=0.5*np.abs(mp-mc)
        finite_repro=repro_obs[np.isfinite(repro_obs)]
        common_repro=float(np.median(finite_repro)) if finite_repro.size else 0.0
        repro=np.where(np.isfinite(repro_obs),repro_obs,common_repro)
        sigma=np.sqrt(stat**2 + repro**2)
        if np.any(~np.isfinite(sigma)):
            raise RuntimeError(f"{field}: primary scan missing target groups; sigma={sigma}")
        draws=RNG.normal(targets, sigma, size=(N_MC,4))
        pabs=(D_GHZ-draws)*1000.0/GAMMA_MHZ_G
        proj=pabs*signs[None,:]
        bdraw=proj @ pinv.T
        bmag=np.linalg.norm(bdraw,axis=1)
        bref=np.asarray(cfg["B"],float)
        cos=np.sum(bdraw*bref[None,:],axis=1)/(bmag*np.linalg.norm(bref))
        ang=np.degrees(np.arccos(np.clip(cos,-1,1)))

        bp=solve_from_axis_freqs(mp,signs)
        companion_complete=bool(np.all(np.isfinite(mc)))
        bc=solve_from_axis_freqs(mc,signs) if companion_complete else np.full(3,np.nan)
        companion_delta=float(np.linalg.norm(bc-bref)) if companion_complete else np.nan
        companion_angle=angle_deg(bc,bref) if companion_complete else np.nan
        row=dict(
            field=field,
            Bx_ref_G=bref[0],By_ref_G=bref[1],Bz_ref_G=bref[2],
            Bmag_ref_G=float(np.linalg.norm(bref)),
            sigma_Bx_G=float(np.std(bdraw[:,0],ddof=1)),
            sigma_By_G=float(np.std(bdraw[:,1],ddof=1)),
            sigma_Bz_G=float(np.std(bdraw[:,2],ddof=1)),
            sigma_Bmag_G=float(np.std(bmag,ddof=1)),
            angle68_deg=float(np.percentile(ang,68)),
            angle95_deg=float(np.percentile(ang,95)),
            primary_deltaB_G=float(np.linalg.norm(bp-bref)),
            primary_angle_deg=angle_deg(bp,bref),
            companion_branches=int(np.sum(np.isfinite(mc))),
            companion_deltaB_G=companion_delta,
            companion_angle_deg=companion_angle,
            common_repro_halfdiff_MHz=common_repro*1000,
            max_line_sigma_MHz=float(np.max(sigma)*1000),
        )
        brows.append(row)
        details[field]=dict(
            B_ref_G=bref.tolist(),
            targets_GHz=targets.tolist(),
            signs=signs.tolist(),
            primary_medians_GHz=mp.tolist(),
            companion_medians_GHz=mc.tolist(),
            primary_bootstrap_se_MHz=(sep*1000).tolist(),
            companion_bootstrap_se_MHz=(sec*1000).tolist(),
            primary_robust_spread_MHz=(madp*1000).tolist(),
            companion_robust_spread_MHz=(madc*1000).tolist(),
            primary_n=npop.tolist(), companion_n=nc.tolist(),
            reproducibility_halfdiff_MHz=(repro*1000).tolist(),
            adopted_sigma_line_MHz=(sigma*1000).tolist(),
            B_from_primary_G=bp.tolist(), B_from_companion_G=bc.tolist(),
            covariance_B_G2=np.cov(bdraw,rowvar=False).tolist(),
            note=cfg["note"],
        )

    ldf=pd.DataFrame(all_rows)
    bdf=pd.DataFrame(brows)
    ldf.to_csv(OUT/"v39_target_guided_line_summary.csv",index=False)
    bdf.to_csv(OUT/"v39_bfield_uncertainty.csv",index=False)
    with open(OUT/"v39_details.json","w",encoding="utf-8") as fh:
        json.dump(details,fh,indent=2)
    lines=[
        "V39 TARGET-GUIDED JOHNSON B-FIELD UNCERTAINTY",
        "============================================",
        "",
        "Method:",
        "  Fit each NV with the optimized two-Voigt ESR fitter.",
        "  Keep only fits with f1 < D < f2.",
        f"  Assign f1 to the nearest established crystallographic branch with non-overlapping adaptive windows capped at {MAX_WINDOW_GHZ*1000:.1f} MHz.",
        "  Estimate each branch center with the median and bootstrap its median.",
        "  Use half the primary/companion center difference as a reproducibility term.",
        "  Combine bootstrap/population SE and reproducibility in quadrature.",
        "  Propagate those line uncertainties through the fixed-sign crystal-frame B solve.",
        "",
        "Important: the historical B vector remains the central value; empirical fits set uncertainty.",
        "The raw-fit medians are diagnostics and are not silently substituted as new field values.",
        "",
        bdf.to_string(index=False),
    ]
    (OUT/"README_V39.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n=== LINE SUMMARY ===")
    print(ldf.to_string(index=False))
    print("\n=== B-FIELD UNCERTAINTY ===")
    print(bdf.to_string(index=False))
    print("\nSaved to",OUT)

if __name__ == "__main__":
    main()
