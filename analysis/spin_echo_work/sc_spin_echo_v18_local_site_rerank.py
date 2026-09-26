"""V18: local lattice-site reranking for the V15 amplitude exceptions.

Purpose
-------
Test whether V14 found the right frequencies but the wrong lattice representative
(and therefore the wrong kappa) for NV 171, 37, 168, and 85.

Frozen:
  * V14 background (all 9 nuisance/background parameters)
  * V14 model order
  * NV orientation
  * catalog frequencies for every trial lattice site

Fitted for each discrete site combination:
  * one shared visibility/amplitude scale s_NV, with the old s<=3 ceiling removed
  * two phases per selected C13 site

No independent site amplitudes and no continuous frequency motion are allowed.
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
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits

import sc_spin_echo_physical_family_search_v6 as v6

V14_ROOT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v14_beta2_taper0_rerank\2026_09"
)
V15_ROOT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v15_free_site_amplitude_diagnostic\2026_09"
)
DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\spin_echo_v18_local_site_rerank\2026_09"
)

TARGET_NVS = (171, 37, 168, 85)
SCALE_MIN = 0.0
SCALE_MAX_DEFAULT = 10.0
FREQ_TOL_KHZ_DEFAULT = 25.0
POOL_SIZE_DEFAULT = 10
MAX_SUBSTITUTIONS_DEFAULT = 2
def find_v14_file():
    fs = [
        p for p in V14_ROOT.glob("*52G*v14_beta2_taper0_candidate_fits.csv.gz")
        if "smoke" not in str(p).lower()
    ]
    if not fs:
        raise FileNotFoundError(f"No production V14 52G file under {V14_ROOT}")
    return max(fs, key=lambda p: p.stat().st_mtime)


def fit_stats(y, e, pred, k):
    yy = np.asarray(y, float)
    ee = np.maximum(np.asarray(e, float), 1e-12)
    pp = np.asarray(pred, float)
    chi2 = float(np.sum(((yy - pp) / ee) ** 2))
    n = int(yy.size)
    k = int(k)
    red = chi2 / max(1, n-k)
    aic = chi2 + 2.0*k
    aicc = aic + 2.0*k*(k+1)/(n-k-1) if n > k+1 else np.inf
    bic = chi2 + k*np.log(max(n, 2))
    return dict(chi2=chi2, red_chi2=red, aicc=float(aicc), bic=float(bic), npar=k)


def cv_positive(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 2 or np.mean(x) <= 0:
        return 0.0 if x.size else np.nan
    return float(np.std(x) / np.mean(x))
def candidate_model(base, t, bg, sites, scale, phases):
    baseline, contrast, carrier = base.carrier_from_bg(t, bg)
    tt = np.asarray(t, float)
    osc = np.zeros_like(tt)
    for s, (phi0, phi1) in zip(sites, phases):
        amp = float(scale) * float(contrast) * float(s["kappa"]) / 4.0
        f0 = float(s["f0_kHz"]) / 1000.0
        f1 = float(s["f1_kHz"]) / 1000.0
        osc += amp * (
            np.cos(2*np.pi*f0*tt + float(phi0))
            + np.cos(2*np.pi*f1*tt + float(phi1))
        )
    return baseline - contrast*carrier + carrier*osc


def incumbent_info(row):
    n = int(row.model_order)
    out = []
    for j in range(1, n+1):
        out.append(dict(
            slot=j,
            site_id=int(row[f"c13_{j}_site_id"]),
            f0_kHz=float(row[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(row[f"c13_{j}_f1_kHz"]),
            kappa=float(row[f"c13_{j}_kappa"]),
            phi0=float(row[f"c13_{j}_phi0"]),
            phi1=float(row[f"c13_{j}_phi1"]),
        ))
    return out
def diversify_pool(g, incumbent_site, max_n):
    """Keep frequency-nearest entries plus kappa extremes for identifiability."""
    if g.empty:
        return g
    max_n = max(1, int(max_n))
    ids = []

    def add(frame):
        for idx in frame.index:
            if idx not in ids:
                ids.append(idx)
            if len(ids) >= max_n:
                return

    # Incumbent first if present.
    add(g[g.site_id == int(incumbent_site)])
    # Reserve space for both frequency-nearest and kappa-diverse alternatives.
    nearest = g.sort_values(["freq_rms_kHz", "freq_max_kHz"])
    add(nearest.head(max(2, max_n//2)))
    add(g.sort_values("kappa", ascending=False).head(max(2, max_n//3)))
    add(g.sort_values("kappa", ascending=True).head(max(2, max_n//3)))
    # Fill any remaining slots with the next-nearest frequencies.
    add(nearest.head(max_n))
    return g.loc[ids[:max_n]].copy()


def build_slot_pools(base, catalog, row, v15_sites, freq_tol_khz, pool_size):
    ori = base.parse_orientation(row.orientation)
    cat = catalog[catalog.ori.map(tuple) == tuple(ori)].copy()
    inc = incumbent_info(row)
    contrast = float(row.contrast)
    pools = {}
    audit = []
    for info in inc:
        slot = int(info["slot"])
        g = cat.copy()
        g["df0_kHz"] = np.abs(g.f0_kHz.astype(float) - info["f0_kHz"])
        g["df1_kHz"] = np.abs(g.f1_kHz.astype(float) - info["f1_kHz"])
        g["freq_max_kHz"] = np.maximum(g.df0_kHz, g.df1_kHz)
        g["freq_rms_kHz"] = np.sqrt(0.5*(g.df0_kHz**2 + g.df1_kHz**2))
        g = g[g.freq_max_kHz <= float(freq_tol_khz)].copy()

        # Always include the incumbent even if a catalog roundoff falls just outside.
        if not np.any(g.site_id.astype(int) == int(info["site_id"])):
            incrow = cat[cat.site_id.astype(int) == int(info["site_id"])].copy()
            if len(incrow):
                incrow["df0_kHz"] = np.abs(incrow.f0_kHz.astype(float)-info["f0_kHz"])
                incrow["df1_kHz"] = np.abs(incrow.f1_kHz.astype(float)-info["f1_kHz"])
                incrow["freq_max_kHz"] = np.maximum(incrow.df0_kHz, incrow.df1_kHz)
                incrow["freq_rms_kHz"] = np.sqrt(
                    0.5*(incrow.df0_kHz**2 + incrow.df1_kHz**2)
                )
                g = pd.concat([g, incrow], ignore_index=False)

        g = g.drop_duplicates("site_id", keep="first")
        g = diversify_pool(g, info["site_id"], pool_size)

        vv = v15_sites[
            (v15_sites.nv_index == int(row.name))
            & (v15_sites.site_id == int(info["site_id"]))
        ]
        amp_target = float(vv.amp_free.iloc[0]) if len(vv) else np.nan
        records = []
        for _, r in g.iterrows():
            rec = dict(
                slot=slot,
                site_id=int(r.site_id),
                f0_kHz=float(r.f0_kHz),
                f1_kHz=float(r.f1_kHz),
                kappa=float(r.kappa),
                distance_A=float(r.get("distance_A", np.nan)),
                df0_kHz=float(r.df0_kHz),
                df1_kHz=float(r.df1_kHz),
                freq_max_kHz=float(r.freq_max_kHz),
                freq_rms_kHz=float(r.freq_rms_kHz),
                incumbent=bool(int(r.site_id) == int(info["site_id"])),
                incumbent_site_id=int(info["site_id"]),
                incumbent_kappa=float(info["kappa"]),
                kappa_ratio_vs_inc=float(r.kappa)/max(float(info["kappa"]),1e-15),
                phase_seed0=float(info["phi0"]),
                phase_seed1=float(info["phi1"]),
            )
            rec["scale_required_from_v15_amp"] = (
                4.0*amp_target/(contrast*rec["kappa"])
                if np.isfinite(amp_target) and contrast > 0 and rec["kappa"] > 0
                else np.nan
            )
            records.append(rec)
            audit.append(dict(nv_index=int(row.name), **rec))
        pools[slot] = records
    return pools, audit


def enumerate_combos(pools, max_substitutions):
    slots = sorted(pools)
    combos = []
    seen = set()
    for choice in itertools.product(*(pools[s] for s in slots)):
        site_ids = tuple(int(c["site_id"]) for c in choice)
        if len(set(site_ids)) != len(site_ids):
            continue
        subs = sum(not bool(c["incumbent"]) for c in choice)
        if subs > int(max_substitutions):
            continue
        if site_ids in seen:
            continue
        seen.add(site_ids)
        scales_req = [c["scale_required_from_v15_amp"] for c in choice]
        combos.append(dict(
            sites=[dict(c) for c in choice],
            site_ids=site_ids,
            substitution_count=int(subs),
            freq_rms_total=float(np.sqrt(np.mean([c["freq_rms_kHz"]**2 for c in choice]))),
            freq_max_total=float(max(c["freq_max_kHz"] for c in choice)),
            amp_scale_cv_from_v15=cv_positive(scales_req),
            amp_scale_median_from_v15=float(np.nanmedian(scales_req)),
        ))
    combos.sort(key=lambda q: (
        q["substitution_count"],
        np.inf if not np.isfinite(q["amp_scale_cv_from_v15"]) else q["amp_scale_cv_from_v15"],
        q["freq_rms_total"],
    ))
    return combos
def fit_combo(base, t, y, e, bg, combo, scale_seed, scale_max):
    sites = combo["sites"]
    n = len(sites)
    ee = base.safe_err(e)
    phase_seed = np.array(
        [[s["phase_seed0"], s["phase_seed1"]] for s in sites], float
    ).ravel()

    lb = np.r_[SCALE_MIN, np.full(2*n, -np.pi)]
    ub = np.r_[float(scale_max), np.full(2*n, np.pi)]

    def unpack(x):
        return float(x[0]), np.asarray(x[1:], float).reshape(n,2)

    def pred(x):
        scale, phases = unpack(x)
        return candidate_model(base, t, bg, sites, scale, phases)

    def resid(x):
        return (np.asarray(y,float) - pred(x))/ee

    scale_starts = [
        float(scale_seed),
        float(combo.get("amp_scale_median_from_v15", np.nan)),
        1.0, 2.0, 3.0, 4.5,
    ]
    starts = []
    for s in scale_starts:
        if not np.isfinite(s):
            continue
        starts.append(np.r_[np.clip(s, SCALE_MIN+1e-8, scale_max-1e-8), phase_seed])
    starts.append(np.r_[min(3.0, scale_max-1e-8), np.zeros(2*n)])

    # Deterministic random phase starts guard against local phase minima.
    rng_seed = 20260925 + sum((j+1)*int(s["site_id"]) for j,s in enumerate(sites))
    rng = np.random.default_rng(rng_seed)
    for s in (1.5, 3.0, min(5.0, scale_max-1e-6)):
        starts.append(np.r_[s, rng.uniform(-np.pi, np.pi, 2*n)])

    robust = []
    for x0 in starts:
        x0 = np.clip(np.asarray(x0,float), lb+1e-9, ub-1e-9)
        try:
            rr = least_squares(
                resid, x0, bounds=(lb,ub), loss="soft_l1", f_scale=1.0,
                max_nfev=6000, x_scale="jac",
            )
            pp = pred(rr.x)
            chi2 = fit_stats(y,ee,pp,len(rr.x))["chi2"]
            robust.append((chi2, rr.x))
        except Exception:
            pass
    if not robust:
        raise RuntimeError(f"all starts failed for {combo['site_ids']}")
    robust.sort(key=lambda z:z[0])
    finals = []
    for _, x0 in robust[:3]:
        try:
            ff = least_squares(
                resid, x0, bounds=(lb,ub), loss="linear",
                max_nfev=15000, ftol=1e-10, xtol=1e-10, gtol=1e-10,
                x_scale="jac",
            )
            pp = pred(ff.x)
            st = fit_stats(y,ee,pp,len(ff.x))
            finals.append(dict(x=np.asarray(ff.x,float), pred=pp, **st))
        except Exception:
            pass
    if not finals:
        raise RuntimeError(f"final polish failed for {combo['site_ids']}")
    best = min(finals, key=lambda q:(q["chi2"],q["bic"]))
    scale, phases = unpack(best["x"])
    best["scale"] = scale
    best["phases"] = phases
    return best


def combo_to_row(nv, combo, fit):
    out = dict(
        nv_index=int(nv),
        site_key=str(tuple(int(x) for x in combo["site_ids"])),
        substitution_count=int(combo["substitution_count"]),
        freq_rms_total=float(combo["freq_rms_total"]),
        freq_max_total=float(combo["freq_max_total"]),
        amp_scale_cv_from_v15=float(combo["amp_scale_cv_from_v15"]),
        amp_scale_median_from_v15=float(combo["amp_scale_median_from_v15"]),
        shared_scale=float(fit["scale"]),
        chi2=float(fit["chi2"]),
        red_chi2=float(fit["red_chi2"]),
        aicc=float(fit["aicc"]),
        bic=float(fit["bic"]),
        npar=int(fit["npar"]),
        phases_json=json.dumps(np.asarray(fit["phases"],float).tolist()),
    )
    for j, s in enumerate(combo["sites"],1):
        out[f"c13_{j}_site_id"] = int(s["site_id"])
        out[f"c13_{j}_f0_kHz"] = float(s["f0_kHz"])
        out[f"c13_{j}_f1_kHz"] = float(s["f1_kHz"])
        out[f"c13_{j}_kappa"] = float(s["kappa"])
        out[f"c13_{j}_kappa_ratio_vs_inc"] = float(s["kappa_ratio_vs_inc"])
        out[f"c13_{j}_freq_rms_kHz"] = float(s["freq_rms_kHz"])
        out[f"c13_{j}_incumbent"] = bool(s["incumbent"])
    return out


def config_tag(args):
    def fnum(x):
        return (f"{float(x):g}").replace(".", "p")
    return (
        f"tol{fnum(args.freq_tol_khz)}_pool{int(args.pool_size)}_"
        f"sub{int(args.max_substitutions)}_smax{fnum(args.scale_max)}"
    )


def fit_one_nv(base, t, y, e, row, catalog, v15_sites, args, outdir):
    nv = int(row.name)
    tag = config_tag(args)
    checkpoint = outdir / f"checkpoint_nv_{nv:04d}_{tag}.csv.gz"
    pool_path = outdir / f"candidate_pool_nv_{nv:04d}_{tag}.csv"
    pools, audit = build_slot_pools(
        base, catalog, row, v15_sites,
        args.freq_tol_khz, args.pool_size,
    )
    pd.DataFrame(audit).to_csv(pool_path, index=False)

    combos = enumerate_combos(pools, args.max_substitutions)
    if args.max_combos and len(combos) > int(args.max_combos):
        incumbent = [q for q in combos if q["substitution_count"] == 0]
        others = [q for q in combos if q["substitution_count"] > 0]
        others.sort(key=lambda q: (
            np.inf if not np.isfinite(q["amp_scale_cv_from_v15"]) else q["amp_scale_cv_from_v15"],
            q["freq_rms_total"],
        ))
        combos = incumbent + others[:max(0,int(args.max_combos)-len(incumbent))]

    print(
        f"NV{nv}: pools={[len(pools[s]) for s in sorted(pools)]}, "
        f"trial combinations={len(combos)}"
    )
    if args.dry_run:
        return dict(nv_index=nv, n_combos=len(combos), dry_run=True)

    if checkpoint.exists() and not args.force:
        print(f"NV{nv}: using existing checkpoint {checkpoint}")
        fits = pd.read_csv(checkpoint)
    else:
        bg = np.asarray(json.loads(row.theta_json),float)[:9]
        scale_seed = float(row.visibility_scale)
        rows = []
        with threadpool_limits(limits=1):
            for i, combo in enumerate(combos,1):
                try:
                    fit = fit_combo(
                        base,t,y,e,bg,combo,scale_seed,args.scale_max
                    )
                    rows.append(combo_to_row(nv,combo,fit))
                except Exception as exc:
                    rows.append(dict(
                        nv_index=nv,
                        site_key=str(tuple(combo["site_ids"])),
                        substitution_count=int(combo["substitution_count"]),
                        fit_error=str(exc),
                        bic=np.inf, chi2=np.inf,
                    ))
                if i % 25 == 0 or i == len(combos):
                    print(f"NV{nv}: {i}/{len(combos)} combinations")
        fits = pd.DataFrame(rows)
        fits.to_csv(checkpoint,index=False,compression="gzip")

    good = fits[np.isfinite(fits.bic)].sort_values("bic").copy()
    if good.empty:
        raise RuntimeError(f"NV{nv}: no successful V18 fits")

    inc = good[good.substitution_count == 0].sort_values("bic").iloc[0]
    good["delta_bic_vs_incumbent"] = good.bic - float(inc.bic)
    good["rank_bic"] = np.arange(1,len(good)+1)
    ranked_path = outdir / f"ranked_nv_{nv:04d}_{tag}.csv"
    good.to_csv(ranked_path,index=False)
    best = good.iloc[0]
    summary = dict(
        nv_index=nv,
        n_combos=int(len(good)),
        incumbent_site_key=str(inc.site_key),
        incumbent_scale=float(inc.shared_scale),
        incumbent_bic=float(inc.bic),
        best_site_key=str(best.site_key),
        best_substitution_count=int(best.substitution_count),
        best_scale=float(best.shared_scale),
        best_bic=float(best.bic),
        delta_bic_best_vs_incumbent=float(best.bic-inc.bic),
        site_assignment_changed=bool(str(best.site_key) != str(inc.site_key)),
        strong_site_evidence=bool(float(best.bic-inc.bic) <= -6.0),
        decisive_site_evidence=bool(float(best.bic-inc.bic) <= -10.0),
        pool_file=str(pool_path),
        ranked_file=str(ranked_path),
    )
    return summary


def reconstruct_curve(base,t,row,fitrow):
    bg=np.asarray(json.loads(row.theta_json),float)[:9]
    n=int(row.model_order)
    phases=np.asarray(json.loads(fitrow.phases_json),float)
    sites=[]
    for j in range(1,n+1):
        sites.append(dict(
            site_id=int(fitrow[f"c13_{j}_site_id"]),
            f0_kHz=float(fitrow[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(fitrow[f"c13_{j}_f1_kHz"]),
            kappa=float(fitrow[f"c13_{j}_kappa"]),
        ))
    return candidate_model(
        base,t,bg,sites,float(fitrow.shared_scale),phases
    )


def make_pdf(base,t,Y,E,winners,summary,outdir):
    pdf_path=outdir/"v18_52G_local_site_rerank.pdf"
    with PdfPages(pdf_path) as pdf:
        for _,s in summary.sort_values("nv_index").iterrows():
            nv=int(s.nv_index)
            ranked=pd.read_csv(Path(s.ranked_file))
            inc=ranked[ranked.substitution_count==0].sort_values("bic").iloc[0]
            best=ranked.sort_values("bic").iloc[0]
            row=winners.loc[nv]
            c0=reconstruct_curve(base,t,row,inc)
            cb=reconstruct_curve(base,t,row,best)

            fig,axes=plt.subplots(2,1,figsize=(10,7.5),height_ratios=[2.1,1.0])
            ax=axes[0]
            ax.errorbar(t,Y[nv],yerr=base.safe_err(E[nv]),fmt="o",ms=3,capsize=1,label="data")
            ax.plot(t,c0,lw=1.2,label=f"incumbent {inc.site_key}")
            ax.plot(t,cb,lw=1.5,label=f"best {best.site_key}")
            ax.set(xlabel="Total Hahn-echo evolution time (us)",ylabel="Normalized signal")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            ax.set_title(
                f"NV {nv} | dBIC(best-inc)={best.bic-inc.bic:+.1f} | "
                f"s={best.shared_scale:.2f}"
            )
            ax=axes[1]
            top=ranked.head(12).copy()
            labels=[str(x) for x in top.site_key]
            vals=top.bic-float(inc.bic)
            ax.bar(np.arange(len(top)),vals)
            ax.axhline(0,ls="--",lw=.8)
            ax.set_xticks(np.arange(len(top)),labels,rotation=60,ha="right",fontsize=7)
            ax.set(ylabel="dBIC vs incumbent",xlabel="Site combination")
            ax.grid(axis="y",alpha=.2)
            fig.tight_layout()
            pdf.savefig(fig,bbox_inches="tight")
            plt.close(fig)
    return pdf_path


def run(args):
    outdir=Path(args.output_dir)
    outdir.mkdir(parents=True,exist_ok=True)

    base=v6.load_backend("52G")
    _,checkpoint,_,_=base.discover_paths()
    t,Y,E=base.load_data(checkpoint)

    v14_path=find_v14_file()
    d=pd.read_csv(v14_path)
    winners=d[d.rank_global_bic==1].set_index("nv_index")
    catalog=v6.load_catalog(base)
    v15_sites=pd.read_csv(V15_ROOT/"v15_52G_site_amplitudes.csv")

    targets=[int(x) for x in args.nv.split(",") if x.strip()]
    missing=[nv for nv in targets if nv not in winners.index]
    if missing:
        raise KeyError(f"Targets missing from V14 winners: {missing}")

    if args.dry_run:
        summaries=[
            fit_one_nv(
                base,t,Y[nv],E[nv],winners.loc[nv],catalog,v15_sites,args,outdir
            )
            for nv in targets
        ]
        print(pd.DataFrame(summaries).to_string(index=False))
        return

    def job(nv):
        return fit_one_nv(
            base,t,Y[nv],E[nv],winners.loc[nv],catalog,v15_sites,args,outdir
        )

    summaries=Parallel(
        n_jobs=max(1,min(int(args.workers),len(targets))),
        backend="loky",verbose=10,
    )(delayed(job)(nv) for nv in targets)

    summary=pd.DataFrame(summaries).sort_values("nv_index")
    summary_path=outdir/"v18_52G_summary.csv"
    summary.to_csv(summary_path,index=False)
    pdf_path=make_pdf(base,t,Y,E,winners,summary,outdir)

    meta=dict(
        source_v14=str(v14_path),
        targets=targets,
        freq_tol_khz=float(args.freq_tol_khz),
        pool_size=int(args.pool_size),
        max_substitutions=int(args.max_substitutions),
        scale_max=float(args.scale_max),
        max_combos=int(args.max_combos) if args.max_combos else None,
        workers=int(args.workers),
        model=(
            "V14 background/order/orientation fixed; discrete local catalog sites; "
            "one shared scale; free phases; no continuous frequency fit"
        ),
    )
    with open(outdir/"v18_run_metadata.json","w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    print("\nV18 COMPLETE")
    print(summary.to_string(index=False))
    print("\nSummary:",summary_path)
    print("PDF:",pdf_path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nv",default=",".join(str(x) for x in TARGET_NVS))
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    ap.add_argument("--freq-tol-khz",type=float,default=FREQ_TOL_KHZ_DEFAULT)
    ap.add_argument("--pool-size",type=int,default=POOL_SIZE_DEFAULT)
    ap.add_argument(
        "--max-substitutions",type=int,choices=[1,2,3],
        default=MAX_SUBSTITUTIONS_DEFAULT,
    )
    ap.add_argument("--scale-max",type=float,default=SCALE_MAX_DEFAULT)
    ap.add_argument(
        "--max-combos",type=int,default=0,
        help="0 means no cap after local pool/substitution filtering",
    )
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--dry-run",action="store_true")
    ap.add_argument(
        "--force",action="store_true",
        help="refit even if per-NV checkpoint already exists",
    )
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
