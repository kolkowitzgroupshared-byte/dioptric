"""V36 magnetic-field vector sensitivity audit.
Quantifies how ODMR line uncertainty propagates into reconstructed B.
This is a sensitivity curve, not an asserted experimental error bar.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"analysis"/"b_field_and_coils_calcaultions"))
from sc_b_field_calculations import solve_B_from_odmr_order_invariant, nv_axes

OUT=Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v36_bfield_uncertainty\2026_09")
OUT.mkdir(parents=True,exist_ok=True)
D=2.8785; GAM=2.8025
FIELDS={
"49G":np.array([-46.19581364,-17.44900422,-5.57935388]),
"59G":np.array([-41.57848995,-32.77145194,-27.5799348]),
"62G":np.array([-48.67047318,-32.07615947,22.49657427]),
"65G":np.array([-31.61263115,-56.58135644,-6.5512002])}
axes=nv_axes()
sigmas=[0.05,0.10,0.25,0.50] # MHz per ODMR line
rng=np.random.default_rng(20260929)
rows=[]
for lab,B in FIELDS.items():
    f=D-GAM*np.abs(axes@B)/1000.0
    Bref=np.asarray(B,float).copy()
    # global sign is physically equivalent for this |projection|-only reconstruction
    def align(x):
        return x if np.linalg.norm(x-Bref)<=np.linalg.norm(-x-Bref) else -x
    for sig in sigmas:
        arr=[]
        for _ in range(3000):
            fp=f+rng.normal(0,sig/1000,4)
            bmag=(D-fp)*1000.0/GAM
            sgn=np.sign(axes@Bref); sgn[sgn==0]=1
            Bp=np.linalg.lstsq(axes,sgn*bmag,rcond=None)[0]
            arr.append(align(Bp))
        A=np.asarray(arr); mags=np.linalg.norm(A,axis=1)
        bh=A/mags[:,None]; br=Bref/np.linalg.norm(Bref)
        ang=np.degrees(np.arccos(np.clip(np.abs(bh@br),-1,1)))
        rows.append({"field":lab,"line_sigma_MHz":sig,"Bmag_nom_G":np.linalg.norm(Bref),
          "Bmag_std_G":np.std(mags),"angle_median_deg":np.median(ang),
          "angle_p68_deg":np.quantile(ang,.68),"angle_p95_deg":np.quantile(ang,.95),
          "Bx_std_G":np.std(A[:,0]),"By_std_G":np.std(A[:,1]),"Bz_std_G":np.std(A[:,2])})
df=pd.DataFrame(rows)
df.to_csv(OUT/"v36_odmr_to_bfield_sensitivity.csv",index=False)
fig,ax=plt.subplots(figsize=(7,4.8))
for lab,g in df.groupby("field"):
    ax.plot(g.line_sigma_MHz,g.angle_p68_deg,marker="o",label=lab)
ax.set_xlabel("Assumed 1-sigma uncertainty per ODMR line (MHz)")
ax.set_ylabel("68% B-direction error (deg)")
ax.set_title("ODMR-to-B direction sensitivity")
ax.legend(); fig.tight_layout()
fig.savefig(OUT/"01_direction_sensitivity.png",dpi=250); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.8))
for lab,g in df.groupby("field"):
    ax.plot(g.line_sigma_MHz,g.Bmag_std_G,marker="o",label=lab)
ax.set_xlabel("Assumed 1-sigma uncertainty per ODMR line (MHz)")
ax.set_ylabel("1-sigma |B| uncertainty (G)")
ax.set_title("ODMR-to-field-magnitude sensitivity")
ax.legend(); fig.tight_layout()
fig.savefig(OUT/"02_magnitude_sensitivity.png",dpi=250); plt.close(fig)
meta={"interpretation":"Sensitivity transfer function only. Experimental confidence intervals require the actual ODMR fit covariance/residuals used to reconstruct each nominal B vector.",
"nominal_fields_G":{k:v.tolist() for k,v in FIELDS.items()},
"D_GHz":D,"gamma_e_MHz_per_G":GAM,"draws_per_point":3000}
with open(OUT/"v36_metadata.json","w") as f: json.dump(meta,f,indent=2)
print(df.to_string(index=False))
print("OUTPUT",OUT)
