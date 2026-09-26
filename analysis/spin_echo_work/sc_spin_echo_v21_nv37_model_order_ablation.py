"""V21: NV37 model-order ablation on the validated spectral sites.

Tests every non-empty subset of (31,24,78), with full V14 reduced-background
reoptimization (beta=2, taper=0), under two amplitude laws:

  shared_physical:
      one shared q multiplying C*kappa_j/4 for all included sites

  free_site_equal:
      one q_j per included site, but equal f+ and f- amplitude within each site

Frequencies are fixed to the catalog and phases are fitted.
This distinguishes over-selection of a weak site from failure of the shared
site-amplitude law.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v20_nv37_sideband_weight_diagnostic as v20

NV=37
FULL_SITES=(31,24,78)
V19_FILE=v20.V19_FILE
DEFAULT_OUT=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v21_nv37_model_order\2026_09"
)
def subsets():
    out=[]
    for n in (3,2,1):
        for c in itertools.combinations(FULL_SITES,n):
            out.append(tuple(c))
    return out


def phase_seed_map(v19_theta):
    th=np.asarray(v19_theta,float)
    ph=th[10:].reshape(3,2)
    return {sid:ph[i].copy() for i,sid in enumerate(FULL_SITES)}


def load_sites(base,orientation,ids):
    d=v6.load_catalog(base)
    idx={(tuple(r.ori),int(r.site_id)):r for r in d.itertuples()}
    out=[]
    for sid in ids:
        r=idx[(tuple(orientation),int(sid))]
        out.append(dict(
            site_id=int(sid),f0_kHz=float(r.f0_kHz),f1_kHz=float(r.f1_kHz),
            kappa=float(r.kappa),distance_A=float(getattr(r,"distance_A",np.nan)),
        ))
    return out


def fit_subset(base,t,y,e,bg,scale,phmap,sites,amp_model,baseline_bound_seed):
    ids=tuple(int(s["site_id"]) for s in sites)
    ph=np.asarray([phmap[sid] for sid in ids],float)
    kind="M0_equal" if amp_model=="shared_physical" else "Msite_free_equal"
    fit=v20.fit_kind(
        base,t,y,e,sites,kind,bg,scale,ph,
        baseline_bound_seed=baseline_bound_seed,
    )
    rec=v20.result_record(base,t,kind,fit,sites)
    rec.update(
        amp_model=amp_model,
        model_order=len(ids),
        site_key=str(ids),
    )
    return rec,fit["pred"]


def summarize(tab):
    t=tab.copy()
    t["rank_bic"]=t.bic.rank(method="first").astype(int)
    for amp,g in t.groupby("amp_model"):
        b=float(g.bic.min())
        t.loc[g.index,"delta_bic_within_amp_best"]=g.bic-b

    # Comparisons centered on the exact question: does dropping site 24 help?
    rows=[]
    for amp in t.amp_model.unique():
        g=t[t.amp_model==amp].set_index("site_key")
        full=g.loc[str(FULL_SITES)]
        drop24=g.loc[str((31,78))]
        rows.append(dict(
            amp_model=amp,
            full_site_key=str(FULL_SITES),
            drop24_site_key=str((31,78)),
            full_bic=float(full.bic),
            drop24_bic=float(drop24.bic),
            delta_bic_drop24_minus_full=float(drop24.bic-full.bic),
            full_chi2=float(full.chi2),
            drop24_chi2=float(drop24.chi2),
            delta_chi2_drop24_minus_full=float(drop24.chi2-full.chi2),
            prefer_drop24=bool(drop24.bic<full.bic),
        ))
    return t,pd.DataFrame(rows)
def plot_pdf(pdf,t,y,e,tab,preds):
    for amp in ("shared_physical","free_site_equal"):
        g=tab[tab.amp_model==amp].sort_values("bic")
        fig,axes=plt.subplots(2,1,figsize=(10,7.5),height_ratios=[2.1,1.0])
        ax=axes[0]
        ax.errorbar(t,y,yerr=e,fmt="o",ms=3,capsize=1,label="data")
        for _,r in g.head(4).iterrows():
            key=(amp,r.site_key)
            ax.plot(t,preds[key],lw=1.2,label=f"{r.site_key} BIC={r.bic:.1f}")
        ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Normalized signal")
        ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2)
        ax.set_title(f"NV37 model-order ablation: {amp}")

        ax=axes[1]
        order=g.sort_values(["model_order","bic"],ascending=[False,True])
        x=np.arange(len(order))
        vals=order.bic-float(g.bic.min())
        ax.bar(x,vals)
        ax.set_xticks(x,[f"N{int(n)} {k}" for n,k in zip(order.model_order,order.site_key)],
                      rotation=45,ha="right",fontsize=8)
        ax.set(ylabel="delta BIC from best",xlabel="Site subset")
        ax.grid(axis="y",alpha=.2)
        fig.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)


def run(args):
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    base=v6.load_backend("52G")
    _,ck,_,_=base.discover_paths(); t,Y,E=base.load_data(ck)

    v19=pd.read_csv(V19_FILE)
    q=v19[(v19.nv_index==NV)&(v19.hypothesis=="H0")].iloc[0]
    th=np.asarray(json.loads(q.theta_json),float)
    bg=th[:9]; scale=float(q.shared_scale); phmap=phase_seed_map(th)

    v14root=Path(
        r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v14_beta2_taper0_rerank\2026_09"
    )
    vf=max(v14root.glob("*52G*v14_beta2_taper0_candidate_fits.csv.gz"),
           key=lambda p:p.stat().st_mtime)
    vd=pd.read_csv(vf)
    row=vd[(vd.rank_global_bic==1)&(vd.nv_index==NV)].iloc[0]
    ori=base.parse_orientation(row.orientation)
    baseline_bound_seed=float(row.baseline)

    records=[]; preds={}
    with threadpool_limits(limits=1):
        for amp_model in ("shared_physical","free_site_equal"):
            for ids in subsets():
                sites=load_sites(base,ori,ids)
                rec,pred=fit_subset(
                    base,t,Y[NV],E[NV],bg,scale,phmap,sites,amp_model,
                    baseline_bound_seed,
                )
                records.append(rec); preds[(amp_model,str(ids))]=pred
                print(
                    f"{amp_model:15s} {ids}: N={len(ids)} "
                    f"chi2={rec['chi2']:.3f} BIC={rec['bic']:.3f} "
                    f"redchi2={rec['red_chi2']:.3f}"
                )

    tab=pd.DataFrame(records)
    ranked,comparison=summarize(tab)
    ranked=ranked.sort_values("bic")
    ranked.to_csv(outdir/"v21_nv37_model_order_fits.csv",index=False)
    comparison.to_csv(outdir/"v21_nv37_drop24_comparison.csv",index=False)

    with PdfPages(outdir/"v21_nv37_model_order_ablation.pdf") as pdf:
        plot_pdf(pdf,t,Y[NV],base.safe_err(E[NV]),ranked,preds)
    print("\nGLOBAL BIC RANKING")
    print(ranked[[
        "amp_model","model_order","site_key","npar","chi2","red_chi2","bic","rank_bic"
    ]].to_string(index=False))
    print("\nDROP-SITE-24 TEST")
    print(comparison.to_string(index=False))
    print("\nSaved:",outdir)
    return ranked,comparison


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
