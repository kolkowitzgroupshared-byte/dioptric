"""V16 targeted diagnostic for the four V15 site-amplitude exceptions.

V14 background, selected C13 sites, catalog frequencies and fitted sideband
phases stay fixed.  Compare V15 additive models with exact single-spin Hahn
terms, both summed and multiplied.  Test both the total-evolution convention
(arg_scale=0.5) and the catalog-sideband convention (arg_scale=1.0).
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
from scipy.optimize import least_squares

import sc_c13_spin_echo_physical_family_search_v6 as v6
import sc_c13_spin_echo_v15_free_site_amplitude_diagnostic as v15

TARGET_NVS = (171, 37, 168, 85)
DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo"
    r"\spin_echo_v16_product_cross_term_diagnostic\2026_09"
)
SCALE_BOUNDS = (0.0, 8.0)
DT_BOUNDS_US = (-5.0, 5.0)
def phase_stats(y, e, pred, k):
    return v15.conditional_stats(y, e, pred, k)


def q_site(t_us, site, dt_us, arg_scale):
    """Exact single-spin Hahn modulation depth for one catalog site."""
    fplus = max(float(site["f0_kHz"]), float(site["f1_kHz"])) / 1000.0
    fminus = min(float(site["f0_kHz"]), float(site["f1_kHz"])) / 1000.0
    fI = 0.5 * (fplus - fminus)
    fm = 0.5 * (fplus + fminus)
    tt = np.asarray(t_us, float) + float(dt_us)
    a = float(arg_scale) * np.pi
    return 2.0 * float(site["kappa"]) * (
        np.sin(a * fI * tt) ** 2
    ) * (
        np.sin(a * fm * tt) ** 2
    )


def physical_basis(t, contrast, carrier, sites, dts, arg_scale, combine):
    qs = np.vstack([
        q_site(t, site, dt, arg_scale)
        for site, dt in zip(sites, dts)
    ])
    if combine == "sum":
        mod = np.sum(qs, axis=0)
    elif combine == "product":
        mod = 1.0 - np.prod(1.0 - qs, axis=0)
    else:
        raise ValueError(combine)
    return float(contrast) * np.asarray(carrier, float) * mod
def fit_variant(t, y, e, contrast, carrier, core, sites,
                arg_scale, combine, phase_mode, scale_seed, rng):
    n = len(sites)
    ndt = 1 if phase_mode == "shared_dt" else n
    lb = np.r_[SCALE_BOUNDS[0], np.full(ndt, DT_BOUNDS_US[0])]
    ub = np.r_[SCALE_BOUNDS[1], np.full(ndt, DT_BOUNDS_US[1])]
    ee = np.asarray(e, float)

    def expand_dt(p):
        return np.full(n, p[1]) if ndt == 1 else np.asarray(p[1:], float)

    def predict(p):
        b = physical_basis(
            t, contrast, carrier, sites, expand_dt(p), arg_scale, combine
        )
        return np.asarray(core, float) + float(p[0]) * b

    def residual(p):
        return (np.asarray(y, float) - predict(p)) / ee

    starts = []
    for dt0 in (-4, -2, -1, 0, 1, 2, 4):
        starts.append(np.r_[np.clip(scale_seed, 0.1, 6.0), np.full(ndt, dt0)])
    for _ in range(18 if ndt > 1 else 8):
        starts.append(np.r_[
            rng.uniform(0.5, 5.5),
            rng.uniform(-4.5, 4.5, size=ndt),
        ])
    best = None
    for p0 in starts:
        try:
            q = least_squares(
                residual, np.clip(p0, lb + 1e-9, ub - 1e-9),
                bounds=(lb, ub), loss="linear", max_nfev=6000,
                ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
            )
            pred = predict(q.x)
            st = phase_stats(y, ee, pred, 1 + ndt)
            rec = dict(params=q.x, pred=pred, **st)
            if best is None or rec["chi2"] < best["chi2"]:
                best = rec
        except Exception:
            pass
    if best is None:
        raise RuntimeError(
            f"fit failed: scale={arg_scale}, combine={combine}, mode={phase_mode}"
        )
    p = np.asarray(best.pop("params"), float)
    dts = np.full(n, p[1]) if ndt == 1 else p[1:]
    best["scale"] = float(p[0])
    best["dts_us"] = np.asarray(dts, float)
    best["arg_scale"] = float(arg_scale)
    best["combine"] = str(combine)
    best["phase_mode"] = str(phase_mode)
    best["npar_conditional"] = int(1 + ndt)
    return best
def fit_nv(base, t, y, e, row):
    v15_summary, site_rows, pred_saved, pred_shared, pred_free = (
        v15.fit_one_nv(base, t, y, e, row)
    )
    theta, contrast, carrier, core, sites, _ = v15.frozen_components(base, t, row)
    nv = int(row["nv_index"])
    rng = np.random.default_rng(16000 + nv)
    scale_seed = max(0.5, float(v15_summary["shared_refit_scale"]))

    fits = []
    for arg_scale, convention in ((0.5, "total_evolution"), (1.0, "catalog_sideband")):
        for combine in ("sum", "product"):
            for phase_mode in ("shared_dt", "site_dt"):
                q = fit_variant(
                    t, y, e, contrast, carrier, core, sites,
                    arg_scale, combine, phase_mode, scale_seed, rng,
                )
                q["model"] = f"{combine}_{convention}_{phase_mode}"
                q["nv_index"] = nv
                q["delta_bic_vs_add_shared"] = (
                    q["bic"] - float(v15_summary["shared_refit_bic"])
                )
                q["delta_bic_vs_add_free"] = (
                    q["bic"] - float(v15_summary["free_bic"])
                )
                fits.append(q)
    comp = []
    comp.append(dict(
        nv_index=nv, model="additive_shared_v15",
        chi2=float(v15_summary["shared_refit_chi2"]),
        red_chi2=float(v15_summary["shared_refit_red_chi2"]),
        bic=float(v15_summary["shared_refit_bic"]),
        aicc=float(v15_summary["shared_refit_aicc"]),
        npar_conditional=1,
        scale=float(v15_summary["shared_refit_scale"]),
        delta_bic_vs_add_shared=0.0,
        delta_bic_vs_add_free=(
            float(v15_summary["shared_refit_bic"]) - float(v15_summary["free_bic"])
        ),
    ))
    comp.append(dict(
        nv_index=nv, model="additive_free_v15",
        chi2=float(v15_summary["free_chi2"]),
        red_chi2=float(v15_summary["free_red_chi2"]),
        bic=float(v15_summary["free_bic"]),
        aicc=float(v15_summary["free_aicc"]),
        npar_conditional=int(row["model_order"]),
        scale=np.nan,
        delta_bic_vs_add_shared=float(v15_summary["delta_bic_free_vs_shared"]),
        delta_bic_vs_add_free=0.0,
    ))
    for q in fits:
        comp.append({
            k: q[k] for k in (
                "nv_index", "model", "chi2", "red_chi2", "bic", "aicc",
                "npar_conditional", "scale",
                "delta_bic_vs_add_shared", "delta_bic_vs_add_free",
            )
        })

    dt_rows = []
    for q in fits:
        for j, (site, dt) in enumerate(zip(sites, q["dts_us"]), 1):
            dt_rows.append(dict(
                nv_index=nv, model=q["model"], site_slot=j,
                site_id=int(site["site_id"]), dt_us=float(dt),
                kappa=float(site["kappa"]),
                fplus_kHz=max(site["f0_kHz"], site["f1_kHz"]),
                fminus_kHz=min(site["f0_kHz"], site["f1_kHz"]),
            ))

    curves = {
        "additive_shared_v15": np.asarray(pred_shared, float),
        "additive_free_v15": np.asarray(pred_free, float),
    }
    curves.update({q["model"]: np.asarray(q["pred"], float) for q in fits})
    return pd.DataFrame(comp), pd.DataFrame(dt_rows), curves, v15_summary
def choose_key_models(comp):
    p = comp[comp.model.str.startswith("product_")].copy()
    p_total = p[p.model.str.contains("total_evolution")]
    return (
        str(p.sort_values("bic").iloc[0].model),
        str(p_total.sort_values("bic").iloc[0].model),
    )


def plot_nv(outdir, nv, t, y, e, comp, curves):
    best_product, best_total = choose_key_models(comp)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2.2, 1.0])
    ax = axes[0]
    ax.errorbar(t, y, yerr=e, fmt="o", ms=3, capsize=1, label="data")
    for name, ls in (
        ("additive_shared_v15", "--"),
        ("additive_free_v15", "-"),
        (best_total, ":"),
        (best_product, "-."),
    ):
        ax.plot(t, curves[name], lw=1.4, ls=ls, label=name)
    ax.set(
        xlabel="Stored evolution time (us)", ylabel="Normalized signal",
        title=f"V16 product/cross-term diagnostic | NV {nv}",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    ax = axes[1]
    c = comp.sort_values("bic")
    vals = c["bic"].to_numpy() - float(comp.loc[
        comp.model == "additive_shared_v15", "bic"
    ].iloc[0])
    ax.bar(np.arange(len(c)), vals)
    ax.axhline(0, lw=0.8)
    ax.axhline(-6, lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(c)), c.model, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Delta BIC vs additive shared")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = outdir / f"v16_52G_nv{nv:03d}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def make_summary_plot(outdir, all_comp):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, nv in zip(axes.flat, TARGET_NVS):
        g = all_comp[all_comp.nv_index == nv].sort_values("bic")
        base_bic = float(g.loc[g.model == "additive_shared_v15", "bic"].iloc[0])
        vals = g.bic - base_bic
        ax.bar(np.arange(len(g)), vals)
        ax.axhline(0, lw=0.8)
        ax.axhline(-6, lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(g)), g.model, rotation=70, ha="right", fontsize=6)
        ax.set_title(f"NV {nv}")
        ax.set_ylabel("Delta BIC vs V15 shared")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("V16: fixed-V14 exact Hahn product diagnostic", fontsize=14)
    fig.tight_layout()
    path = outdir / "v16_52G_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    v14_path = v15.find_v14_file("52G")
    cand = pd.read_csv(v14_path)
    winners = cand[cand.rank_global_bic == 1].copy()
    winners = winners[winners.nv_index.isin(TARGET_NVS)].copy()

    base = v6.load_backend("52G")
    _, checkpoint, _, _ = base.discover_paths()
    t, Y, E = base.load_data(checkpoint)
    comps, dts, metadata_rows = [], [], []
    for nv in TARGET_NVS:
        row = winners[winners.nv_index == nv].iloc[0]
        comp, dt, curves, v15_summary = fit_nv(
            base, t, Y[nv], base.safe_err(E[nv]), row
        )
        comps.append(comp)
        dts.append(dt)
        png = plot_nv(
            outdir, nv, t, Y[nv], base.safe_err(E[nv]), comp, curves
        )
        best = comp.sort_values("bic").iloc[0]
        best_product, best_total = choose_key_models(comp)
        metadata_rows.append(dict(
            nv_index=nv,
            v15_delta_bic_free_vs_shared=float(
                v15_summary["delta_bic_free_vs_shared"]
            ),
            best_model=str(best.model),
            best_delta_bic_vs_shared=float(best.delta_bic_vs_add_shared),
            best_product_model=best_product,
            best_total_product_model=best_total,
            plot=str(png),
        ))

    all_comp = pd.concat(comps, ignore_index=True)
    all_dt = pd.concat(dts, ignore_index=True)
    meta_df = pd.DataFrame(metadata_rows)
    all_comp.to_csv(outdir / "v16_52G_model_comparison.csv", index=False)
    all_dt.to_csv(outdir / "v16_52G_site_timing_offsets.csv", index=False)
    meta_df.to_csv(outdir / "v16_52G_key_results.csv", index=False)
    summary_png = make_summary_plot(outdir, all_comp)

    meta = dict(
        source_v14=str(v14_path),
        target_nvs=list(TARGET_NVS),
        scale_bounds=list(SCALE_BOUNDS),
        dt_bounds_us=list(DT_BOUNDS_US),
        physical_total_arg_scale=0.5,
        catalog_sideband_arg_scale=1.0,
        note=(
            "Background/sites/catalog frequencies frozen from V14. "
            "V15 additive phases retained only in additive baselines; "
            "product models use exact Hahn factors with fitted timing offsets."
        ),
        summary_png=str(summary_png),
    )
    with open(outdir / "v16_52G_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nV16 COMPLETE")
    print(meta_df.to_string(index=False))
    print("\nRanked models:")
    for nv in TARGET_NVS:
        g = all_comp[all_comp.nv_index == nv].sort_values("bic")
        print(f"\nNV {nv}")
        print(g[[
            "model", "chi2", "red_chi2", "bic",
            "delta_bic_vs_add_shared", "delta_bic_vs_add_free", "scale"
        ]].to_string(index=False))
    print("\nOutputs:", outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
