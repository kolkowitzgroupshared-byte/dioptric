"""V15 diagnostic: test the V14 shared physical C13 amplitude constraint.

Freeze each V14 BIC winner's background, selected sites, frequencies, and phases.
Compare:
  (1) saved V14 shared amplitude scale,
  (2) best nonnegative shared scale with frozen nuisance parameters,
  (3) independent nonnegative amplitude for every selected C13 site.

No bath/site/frequency/background/phase search is performed here.
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
from scipy.optimize import lsq_linear

import sc_c13_spin_echo_physical_family_search_v6 as v6

V14_ROOT = Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v14_beta2_taper0_rerank\2026_09")
DEFAULT_OUT = Path(r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v15_free_site_amplitude_diagnostic\2026_09")
def find_v14_file(field):
    pattern = f"*{field}*v14_beta2_taper0_candidate_fits.csv.gz"
    files = [p for p in V14_ROOT.glob(pattern) if "smoke" not in str(p).lower()]
    if not files:
        raise FileNotFoundError(f"No V14 candidate file for {field} under {V14_ROOT}")
    return max(files, key=lambda p: p.stat().st_mtime)


def conditional_stats(y, e, pred, k):
    y = np.asarray(y, float)
    e = np.asarray(e, float)
    pred = np.asarray(pred, float)
    chi2 = float(np.sum(((y - pred) / e) ** 2))
    n = int(y.size)
    k = int(k)
    red = chi2 / max(1, n - k)
    aic = chi2 + 2.0 * k
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    bic = chi2 + k * np.log(max(n, 1))
    return dict(chi2=chi2, red_chi2=red, aicc=float(aicc), bic=float(bic))


def site_columns(row, order):
    sites = []
    for j in range(1, order + 1):
        sites.append(dict(
            slot=j,
            site_id=int(row[f"c13_{j}_site_id"]),
            f0_kHz=float(row[f"c13_{j}_f0_kHz"]),
            f1_kHz=float(row[f"c13_{j}_f1_kHz"]),
            kappa=float(row[f"c13_{j}_kappa"]),
            phi0=float(row[f"c13_{j}_phi0"]),
            phi1=float(row[f"c13_{j}_phi1"]),
            distance_A=float(row.get(f"c13_{j}_distance_A", np.nan)),
        ))
    return sites


def frozen_components(base, t, row):
    theta = np.asarray(json.loads(row["theta_json"]), float)
    order = int(row["model_order"])
    bg = theta[:9]
    baseline, contrast, carrier = base.carrier_from_bg(t, bg)
    core = baseline - contrast * carrier
    sites = site_columns(row, order)
    if not sites:
        return theta, contrast, carrier, core, sites, np.empty((len(t), 0))
    cols = []
    tt = np.asarray(t, float)
    for s in sites:
        f0 = s["f0_kHz"] / 1000.0
        f1 = s["f1_kHz"] / 1000.0
        cols.append(carrier * (
            np.cos(2 * np.pi * f0 * tt + s["phi0"]) +
            np.cos(2 * np.pi * f1 * tt + s["phi1"])
        ))
    return theta, contrast, carrier, core, sites, np.column_stack(cols)


def fit_one_nv(base, t, y, e, row):
    nv = int(row["nv_index"])
    order = int(row["model_order"])
    ee = base.safe_err(e)
    theta, contrast, carrier, core, sites, X = frozen_components(base, t, row)

    common = dict(
        nv_index=nv,
        model_order=order,
        site_key=str(row.get("site_key", "")),
        v14_red_chi2=float(row["red_chi2"]),
        v14_bic_full=float(row["bic"]),
        v14_visibility_scale=float(row["visibility_scale"]) if order else np.nan,
        v14_visibility_scale_bound_hit=bool(row.get("visibility_scale_bound_hit", False)),
    )
    if order == 0:
        st = conditional_stats(y, ee, core, 0)
        return common | dict(
            status="N0_no_selected_site",
            shared_saved_chi2=st["chi2"], shared_refit_chi2=st["chi2"],
            free_chi2=st["chi2"], shared_refit_scale=np.nan,
            chi2_gain_global_scale=0.0, chi2_gain_free_vs_shared=0.0,
            delta_bic_free_vs_shared=0.0, delta_aicc_free_vs_shared=0.0,
            free_strong_bic_better=False, shared_scale_exceeds_v14_bound=False,
            r_median=np.nan, r_cv=np.nan, design_condition=np.nan,
        ), [], core, core, core

    scale_saved = float(row["visibility_scale"])
    amp_scale1 = np.array([contrast * s["kappa"] / 4.0 for s in sites], float)
    amp_saved = scale_saved * amp_scale1
    pred_saved = core + X @ amp_saved
    target = np.asarray(y, float) - core
    w = 1.0 / ee
    g = X @ amp_scale1
    gw = g * w
    tw = target * w
    denom = float(gw @ gw)
    scale_refit = max(0.0, float(gw @ tw) / denom) if denom > 0 else 0.0
    pred_shared = core + scale_refit * g

    Xw = X * w[:, None]
    try:
        nn = lsq_linear(Xw, tw, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto")
        amp_free = np.asarray(nn.x, float)
        fit_status = "ok" if nn.success else "lsq_linear_not_converged"
    except Exception as exc:
        amp_free = np.full(order, np.nan)
        fit_status = f"free_fit_error:{type(exc).__name__}"
    pred_free = core + X @ np.nan_to_num(amp_free, nan=0.0)

    # Unconstrained solution is retained only as a phase/model-mismatch diagnostic.
    try:
        amp_ols = np.linalg.lstsq(Xw, tw, rcond=None)[0]
    except Exception:
        amp_ols = np.full(order, np.nan)

    st_saved = conditional_stats(y, ee, pred_saved, 1)
    st_shared = conditional_stats(y, ee, pred_shared, 1)
    st_free = conditional_stats(y, ee, pred_free, order)
    delta_bic = st_free["bic"] - st_shared["bic"]
    delta_aicc = st_free["aicc"] - st_shared["aicc"]
    r = np.divide(amp_free, amp_saved, out=np.full(order, np.nan), where=amp_saved > 0)
    finite_r = r[np.isfinite(r)]
    r_med = float(np.median(finite_r)) if finite_r.size else np.nan
    r_cv = float(np.std(finite_r) / np.mean(finite_r)) if finite_r.size > 1 and np.mean(finite_r) > 0 else np.nan
    try:
        cond = float(np.linalg.cond(Xw))
    except Exception:
        cond = np.nan

    summary = common | dict(
        status=fit_status,
        shared_saved_chi2=st_saved["chi2"],
        shared_saved_red_chi2=st_saved["red_chi2"],
        shared_refit_chi2=st_shared["chi2"],
        shared_refit_red_chi2=st_shared["red_chi2"],
        shared_refit_bic=st_shared["bic"],
        shared_refit_aicc=st_shared["aicc"],
        free_chi2=st_free["chi2"],
        free_red_chi2=st_free["red_chi2"],
        free_bic=st_free["bic"],
        free_aicc=st_free["aicc"],
        shared_refit_scale=scale_refit,
        chi2_gain_global_scale=st_saved["chi2"] - st_shared["chi2"],
        chi2_gain_free_vs_shared=st_shared["chi2"] - st_free["chi2"],
        delta_bic_free_vs_shared=delta_bic,
        delta_aicc_free_vs_shared=delta_aicc,
        free_strong_bic_better=bool(order >= 2 and delta_bic <= -6.0),
        shared_scale_exceeds_v14_bound=bool(scale_refit > float(v6.VISIBILITY_SCALE_MAX)),
        r_median=r_med,
        r_cv=r_cv,
        design_condition=cond,
    )
    site_rows = []
    for idx, s in enumerate(sites):
        af = float(amp_free[idx])
        ap = float(amp_saved[idx])
        ag = float(scale_refit * amp_scale1[idx])
        site_rows.append(dict(
            nv_index=nv, model_order=order, site_slot=s["slot"],
            site_id=s["site_id"], site_key=common["site_key"],
            f0_kHz=s["f0_kHz"], f1_kHz=s["f1_kHz"],
            kappa=s["kappa"], distance_A=s["distance_A"],
            phi0=s["phi0"], phi1=s["phi1"],
            contrast=contrast, v14_scale=scale_saved,
            amp_expected_scale1=float(amp_scale1[idx]),
            amp_v14=ap, amp_shared_refit=ag, amp_free=af,
            amp_ols=float(amp_ols[idx]),
            r_vs_v14=(af / ap if ap > 0 else np.nan),
            r_vs_shared_refit=(af / ag if ag > 0 else np.nan),
            ols_negative=bool(np.isfinite(amp_ols[idx]) and amp_ols[idx] < 0),
        ))
    return summary, site_rows, pred_saved, pred_shared, pred_free


def make_summary_figure(summary, sites):
    s = summary[summary.model_order > 0].copy()
    s2 = s[s.model_order >= 2].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    good = np.isfinite(sites.amp_v14) & np.isfinite(sites.amp_free)
    ax.scatter(sites.loc[good, "amp_v14"], sites.loc[good, "amp_free"], s=16, alpha=0.65)
    if good.any():
        lim = float(np.nanmax([sites.loc[good, "amp_v14"].max(), sites.loc[good, "amp_free"].max()]))
        ax.plot([0, lim], [0, lim], "--", lw=1)
    ax.set(xlabel="V14 predicted site amplitude", ylabel="Free fitted site amplitude",
           title="Per-site amplitude test (all selected sites)")
    ax = axes[0, 1]
    rr = sites.r_vs_v14.replace([np.inf, -np.inf], np.nan).dropna()
    if len(rr):
        hi = min(float(rr.quantile(0.99)), 10.0)
        ax.hist(rr.clip(upper=hi), bins=35)
    ax.axvline(1.0, ls="--", lw=1)
    ax.set(xlabel=r"$r_j=A_j^{free}/A_j^{V14}$", ylabel="Site count",
           title="Amplitude-ratio distribution")

    ax = axes[1, 0]
    if len(s2):
        ax.scatter(s2.v14_red_chi2, s2.chi2_gain_free_vs_shared, s=22, alpha=0.75)
    ax.axhline(0.0, ls="--", lw=1)
    ax.set(xlabel="V14 full-model reduced chi2", ylabel="chi2 gain: shared refit - free sites",
           title="Does site-specific freedom help?")

    ax = axes[1, 1]
    if len(s):
        ax.scatter(s.v14_visibility_scale, s.shared_refit_scale, s=22, alpha=0.75)
        lim = max(float(s.v14_visibility_scale.max()), float(s.shared_refit_scale.max()), 3.0)
        ax.plot([0, lim], [0, lim], "--", lw=1)
        ax.axhline(float(v6.VISIBILITY_SCALE_MAX), ls=":", lw=1)
    ax.set(xlabel="Saved V14 shared scale", ylabel="Frozen-nuisance best shared scale",
           title="Global scale / normalization check")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("V15: fixed-V14 amplitude diagnostic", fontsize=14)
    fig.tight_layout()
    return fig


def diagnostic_page(t, y, e, row, site_rows, pred_saved, pred_shared, pred_free):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2.2, 1.0])
    ax = axes[0]
    ax.errorbar(t, y, yerr=e, fmt="o", ms=3, capsize=1, label="data")
    ax.plot(t, pred_saved, lw=1.2, label="V14 saved shared")
    ax.plot(t, pred_shared, lw=1.2, ls="--", label="shared scale refit")
    ax.plot(t, pred_free, lw=1.5, label="free per-site amplitudes")
    ax.set(xlabel="Total Hahn-echo evolution time (us)", ylabel="Normalized signal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_title(
        f"NV {int(row.nv_index)} | N={int(row.model_order)} | "
        f"dBIC(free-shared)={row.delta_bic_free_vs_shared:.1f} | "
        f"chi2 gain={row.chi2_gain_free_vs_shared:.1f}"
    )

    ax = axes[1]
    sr = site_rows[site_rows.nv_index == int(row.nv_index)].copy()
    if len(sr):
        xx = np.arange(len(sr))
        width = 0.25
        ax.bar(xx - width, sr.amp_v14, width, label="V14")
        ax.bar(xx, sr.amp_shared_refit, width, label="shared refit")
        ax.bar(xx + width, sr.amp_free, width, label="free")
        ax.set_xticks(xx, [str(int(x)) for x in sr.site_id])
    ax.set(xlabel="Selected C13 site id", ylabel="Amplitude")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run(field, outdir, top_pages=24):
    v14_path = find_v14_file(field)
    cand = pd.read_csv(v14_path)
    winners = cand[cand.rank_global_bic == 1].sort_values("nv_index").copy()
    if winners.nv_index.duplicated().any():
        raise RuntimeError("V14 file has duplicate BIC winners for an NV")

    base = v6.load_backend(field)
    _, checkpoint, _, _ = base.discover_paths()
    t, Y, E = base.load_data(checkpoint)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries, site_rows = [], []
    curves_saved, curves_shared, curves_free = {}, {}, {}
    for _, row in winners.iterrows():
        nv = int(row.nv_index)
        summary, sr, ps, pg, pf = fit_one_nv(base, t, Y[nv], E[nv], row)
        summaries.append(summary)
        site_rows.extend(sr)
        curves_saved[nv], curves_shared[nv], curves_free[nv] = ps, pg, pf

    summary = pd.DataFrame(summaries).sort_values("nv_index")
    sites = pd.DataFrame(site_rows)
    summary_path = outdir / f"v15_{field}_nv_summary.csv"
    sites_path = outdir / f"v15_{field}_site_amplitudes.csv"
    summary.to_csv(summary_path, index=False)
    sites.to_csv(sites_path, index=False)

    n_t = len(t)
    n_nv = len(Y)
    cs = np.full((n_nv, n_t), np.nan)
    cg = np.full((n_nv, n_t), np.nan)
    cf = np.full((n_nv, n_t), np.nan)
    for nv in curves_saved:
        cs[nv], cg[nv], cf[nv] = curves_saved[nv], curves_shared[nv], curves_free[nv]
    np.savez_compressed(
        outdir / f"v15_{field}_fit_curves.npz",
        t=np.asarray(t), pred_v14=cs, pred_shared_refit=cg, pred_free=cf,
    )

    fig = make_summary_figure(summary, sites)
    fig.savefig(outdir / f"v15_{field}_summary.png", dpi=180, bbox_inches="tight")
    with PdfPages(outdir / f"v15_{field}_diagnostic.pdf") as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        ranked = summary[summary.model_order >= 2].sort_values(
            ["free_strong_bic_better", "chi2_gain_free_vs_shared"],
            ascending=[False, False],
        )
        for _, row in ranked.head(int(top_pages)).iterrows():
            nv = int(row.nv_index)
            fig = diagnostic_page(
                t, Y[nv], base.safe_err(E[nv]), row, sites,
                curves_saved[nv], curves_shared[nv], curves_free[nv],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    informative = summary[summary.model_order >= 2]
    meta = dict(
        field=field, source_v14=str(v14_path), n_nv=int(len(summary)),
        n_n0=int((summary.model_order == 0).sum()),
        n_n1=int((summary.model_order == 1).sum()),
        n_n2plus=int((summary.model_order >= 2).sum()),
        v14_scale_bound_hits=int(summary.v14_visibility_scale_bound_hit.fillna(False).sum()),
        shared_refit_scale_above_v14_bound=int(summary.shared_scale_exceeds_v14_bound.fillna(False).sum()),
        free_strong_bic_better_n2plus=int(informative.free_strong_bic_better.fillna(False).sum()),
        median_delta_bic_free_vs_shared=float(informative.delta_bic_free_vs_shared.median()),
        median_chi2_gain_free_vs_shared=float(informative.chi2_gain_free_vs_shared.median()),
        median_r_vs_v14=float(sites.r_vs_v14.replace([np.inf, -np.inf], np.nan).median()) if len(sites) else np.nan,
        script_model="V14 background/sites/frequencies/phases frozen; nonnegative amplitudes only",
    )
    with open(outdir / f"v15_{field}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("\nV15 COMPLETE")
    print(json.dumps(meta, indent=2))
    print("\nLargest site-specific improvements (N>=2):")
    cols = ["nv_index", "model_order", "v14_red_chi2", "v14_visibility_scale",
            "shared_refit_scale", "chi2_gain_global_scale", "chi2_gain_free_vs_shared",
            "delta_bic_free_vs_shared", "r_median", "r_cv"]
    print(informative.sort_values("delta_bic_free_vs_shared").head(15)[cols].to_string(index=False))
    print("\nOutputs:")
    print(summary_path)
    print(sites_path)
    print(outdir / f"v15_{field}_diagnostic.pdf")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=["49G", "52G"], default="52G")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--top-pages", type=int, default=24)
    args = ap.parse_args()
    outdir = Path(args.output_dir) if args.output_dir else DEFAULT_OUT
    run(args.field, outdir, args.top_pages)


if __name__ == "__main__":
    main()
