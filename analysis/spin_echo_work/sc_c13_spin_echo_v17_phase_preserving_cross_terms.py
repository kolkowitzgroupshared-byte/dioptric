"""V17: isolate multi-site cross terms without changing V14 frequencies/phases.

The multiplicative model is constructed so its first-order expansion is
identical to the V14/V15 additive sideband model:

    S = b - C*carrier * prod_j(1 - u_j)
    u_j = s_j * kappa_j/4 * [cos(w+ t + phi+) + cos(w- t + phi-)]

Therefore any difference relative to the additive model comes only from
higher-order cross terms between the already-selected V14 sites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares, lsq_linear

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v16_c13_product_diagnostic as v16

DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v17_phase_cross_terms\2026_09"
)
TARGET_NVS = (171, 37, 168, 85)
SCALE_MAX = 6.0
def stats(y, e, pred, k):
    e = np.maximum(np.asarray(e, float), 1e-12)
    chi2 = float(np.sum(((np.asarray(y, float)-np.asarray(pred, float))/e)**2))
    n = len(y); k = int(k)
    red = chi2 / max(1, n-k)
    aic = chi2 + 2*k
    aicc = aic + 2*k*(k+1)/(n-k-1) if n > k+1 else np.inf
    bic = chi2 + k*np.log(max(n, 2))
    return dict(chi2=chi2, red_chi2=red, aicc=float(aicc), bic=float(bic))


def build_h(t, row):
    n = int(row.model_order)
    tt = np.asarray(t, float)
    hs, kappas, site_ids = [], [], []
    for j in range(1, n+1):
        f0 = float(row[f"c13_{j}_f0_kHz"]) / 1000.0
        f1 = float(row[f"c13_{j}_f1_kHz"]) / 1000.0
        p0 = float(row[f"c13_{j}_phi0"])
        p1 = float(row[f"c13_{j}_phi1"])
        hs.append(np.cos(2*np.pi*f0*tt+p0) + np.cos(2*np.pi*f1*tt+p1))
        kappas.append(float(row[f"c13_{j}_kappa"]))
        site_ids.append(int(row[f"c13_{j}_site_id"]))
    return np.asarray(hs, float), np.asarray(kappas, float), site_ids
def cross_model(baseline, contrast, carrier, hs, kappas, scales):
    L = np.ones_like(np.asarray(carrier, float))
    for h, kappa, scale in zip(hs, kappas, scales):
        u = float(scale) * float(kappa) * h / 4.0
        L *= 1.0 - u
    return baseline - contrast * carrier * L


def fit_cross(y, e, baseline, contrast, carrier, hs, kappas, seed_scales,
              free_sites):
    n = len(kappas)
    ee = np.maximum(np.asarray(e, float), 1e-12)
    if free_sites:
        x0 = np.clip(np.asarray(seed_scales, float), 0, SCALE_MAX)
        lb = np.zeros(n); ub = np.full(n, SCALE_MAX)
    else:
        x0 = np.array([float(np.median(seed_scales))])
        lb = np.array([0.0]); ub = np.array([SCALE_MAX])

    def scales(x):
        return np.asarray(x, float) if free_sites else np.full(n, float(x[0]))

    def resid(x):
        pred = cross_model(baseline, contrast, carrier, hs, kappas, scales(x))
        return (np.asarray(y, float)-pred) / ee

    starts = [np.clip(x0, lb+1e-8, ub-1e-8)]
    if not free_sites:
        for s in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
            starts.append(np.array([s]))
    fits = []
    for seed in starts:
        try:
            rr = least_squares(
                resid, seed, bounds=(lb,ub), loss="soft_l1", f_scale=1.0,
                max_nfev=15000, x_scale="jac",
            )
            ff = least_squares(
                resid, rr.x, bounds=(lb,ub), loss="linear",
                max_nfev=25000, ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
            )
            ss = scales(ff.x)
            pred = cross_model(baseline, contrast, carrier, hs, kappas, ss)
            fits.append(dict(scales=ss, pred=pred, **stats(y,e,pred,len(ff.x))))
        except Exception:
            pass
    if not fits:
        raise RuntimeError("cross-term fit failed")
    return min(fits, key=lambda q:(q["chi2"],q["bic"]))


def analyze(base, t, y, e, row):
    n = int(row.model_order)
    theta = np.asarray(json.loads(row.theta_json), float)
    baseline, contrast, carrier = base.carrier_from_bg(t, theta[:9])
    core = baseline - contrast*carrier
    hs, kappas, site_ids = build_h(t,row)
    X = (carrier[None,:] * hs).T
    amp_scale1 = contrast*kappas/4.0
    target = np.asarray(y,float)-core
    ee = base.safe_err(e); w=1.0/ee

    g = X@amp_scale1
    den = float(np.sum((g*w)**2))
    s_shared = max(0.0,float(np.sum((g*w)*(target*w)))/den) if den>0 else 0.0
    pred_add_shared = core+s_shared*g
    st_as = stats(y,ee,pred_add_shared,1)

    nn=lsq_linear(X*w[:,None],target*w,bounds=(0,np.inf))
    amps=np.asarray(nn.x,float)
    s_free=np.divide(amps,amp_scale1,out=np.zeros_like(amps),where=amp_scale1>0)
    pred_add_free=core+X@amps
    st_af=stats(y,ee,pred_add_free,n)

    cr_shared=fit_cross(y,ee,baseline,contrast,carrier,hs,kappas,
                        np.full(n,s_shared),False)
    cr_free=fit_cross(y,ee,baseline,contrast,carrier,hs,kappas,
                      np.clip(s_free,0,SCALE_MAX),True)

    # Evaluate magnitude of the actual cross-term correction at best-fit scales.
    first_order = pred_add_free
    cross_only = cr_free["pred"] - (
        core + X @ (contrast*kappas*cr_free["scales"]/4.0)
    )
    summary=dict(
        nv_index=int(row.name), model_order=n, site_key=str(row.site_key),
        additive_shared_scale=s_shared,
        additive_shared_chi2=st_as["chi2"], additive_shared_bic=st_as["bic"],
        additive_free_chi2=st_af["chi2"], additive_free_bic=st_af["bic"],
        cross_shared_scale=float(cr_shared["scales"][0]),
        cross_shared_chi2=cr_shared["chi2"], cross_shared_bic=cr_shared["bic"],
        cross_free_chi2=cr_free["chi2"], cross_free_bic=cr_free["bic"],
        delta_bic_cross_shared_vs_add_shared=cr_shared["bic"]-st_as["bic"],
        delta_bic_cross_free_vs_add_free=cr_free["bic"]-st_af["bic"],
        chi2_gain_cross_shared=st_as["chi2"]-cr_shared["chi2"],
        chi2_gain_cross_free=st_af["chi2"]-cr_free["chi2"],
        cross_rms=float(np.sqrt(np.mean(cross_only**2))),
        cross_max_abs=float(np.max(np.abs(cross_only))),
    )
    site_rows=[
        dict(nv_index=int(row.name),site_id=sid,kappa=float(kappas[j]),
             additive_free_scale=float(s_free[j]),cross_free_scale=float(cr_free["scales"][j]))
        for j,sid in enumerate(site_ids)
    ]
    curves=dict(add_shared=pred_add_shared,add_free=pred_add_free,
                cross_shared=cr_shared["pred"],cross_free=cr_free["pred"])
    return summary,site_rows,curves
def plot_page(t,y,e,s,sites,curves):
    fig,axes=plt.subplots(2,1,figsize=(10,7.5),height_ratios=[2.2,1.0])
    ax=axes[0]
    ax.errorbar(t,y,yerr=e,fmt="o",ms=3,capsize=1,label="data")
    ax.plot(t,curves["add_shared"],lw=1.1,label="additive shared")
    ax.plot(t,curves["add_free"],lw=1.2,label="additive free")
    ax.plot(t,curves["cross_shared"],lw=1.2,ls="--",label="cross shared")
    ax.plot(t,curves["cross_free"],lw=1.5,label="cross free")
    ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Normalized signal")
    ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=2)
    ax.set_title(
        f"NV {int(s.nv_index)} | dBIC cross-free vs add-free = "
        f"{s.delta_bic_cross_free_vs_add_free:+.1f}"
    )

    ax=axes[1]; x=np.arange(len(sites)); width=.32
    ax.bar(x-width/2,sites.additive_free_scale,width,label="additive free scale")
    ax.bar(x+width/2,sites.cross_free_scale,width,label="cross-model scale")
    ax.set_xticks(x,[str(int(v)) for v in sites.site_id])
    ax.set(xlabel="Selected C13 site id",ylabel="Effective scale")
    ax.grid(axis="y",alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
def run(outdir,targets):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    base=v6.load_backend("52G")
    _,checkpoint,_,_=base.discover_paths()
    t,Y,E=base.load_data(checkpoint)
    d=pd.read_csv(v16.V14_FILE)
    winners=d[d.rank_global_bic==1].set_index("nv_index")

    summaries=[]; site_rows=[]; curves={}
    for nv in targets:
        s,sr,c=analyze(base,t,Y[nv],E[nv],winners.loc[nv])
        summaries.append(s); site_rows.extend(sr); curves[nv]=c
        print(
            f"NV{nv}: dBIC cross-shared={s['delta_bic_cross_shared_vs_add_shared']:+.2f}, "
            f"cross-free={s['delta_bic_cross_free_vs_add_free']:+.2f}"
        )

    sdf=pd.DataFrame(summaries)
    adf=pd.DataFrame(site_rows)
    sdf.to_csv(outdir/"v17_52G_summary.csv",index=False)
    adf.to_csv(outdir/"v17_52G_site_scales.csv",index=False)
    with PdfPages(outdir/"v17_52G_phase_cross_terms.pdf") as pdf:
        for _,s in sdf.sort_values("nv_index").iterrows():
            nv=int(s.nv_index)
            fig=plot_page(t,Y[nv],base.safe_err(E[nv]),s,
                          adf[adf.nv_index==nv],curves[nv])
            pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
    print("\nSUMMARY")
    cols=["nv_index","additive_shared_scale","cross_shared_scale",
          "chi2_gain_cross_shared","delta_bic_cross_shared_vs_add_shared",
          "chi2_gain_cross_free","delta_bic_cross_free_vs_add_free",
          "cross_rms","cross_max_abs"]
    print(sdf[cols].to_string(index=False))
    print("\nSITE SCALES")
    print(adf.to_string(index=False))
    print("\nSaved:",outdir)
    return sdf


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nv",default=",".join(str(x) for x in TARGET_NVS))
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args()
    targets=[int(x) for x in args.nv.split(",") if x.strip()]
    run(args.output_dir,targets)


if __name__=="__main__":
    main()
