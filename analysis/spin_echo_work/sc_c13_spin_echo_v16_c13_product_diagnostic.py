"""V16: fixed-V14-site exact/product 13C diagnostic for the V15 exceptions.

Tests whether the four V15 site-amplitude exceptions are explained by restoring
the full single-13C Hahn-echo coherence product while keeping the V14
background and selected lattice sites fixed.

Models compared on the same data:
  A: additive shared amplitude, unbounded scale (V15 control)
  B: additive free per-site amplitudes (V15 control)
  C: product, one shared scale, dt=0
  D: product, one shared scale + common timing offset dt
  E: product, free per-site scales + common timing offset dt
"""
from __future__ import annotations

import argparse
import ast
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
V14_FILE = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v14_beta2_taper0_rerank\2026_09"
) / (
    "2026_09_21-21_09_41-spin_echo_old_protocol_ranked_52G_v6_physamp_snr0p5_"
    "tol12p0kHz_smax3p0_topS15_perF3_oriassigned_maxC3_v14_beta2_taper0_candidate_fits.csv.gz"
)
CATALOG = Path(r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_52G.json")
V15_SITE = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v15_free_site_amplitude_diagnostic\2026_09"
) / "v15_52G_site_amplitudes.csv"
DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v16_c13_product_diagnostic\2026_09"
)
TARGET_NVS = (171, 37, 168, 85)
SCALE_MAX = 6.0
DT_BOUND_US = 0.35


def canonical_orientation(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            x = [int(z) for z in x.strip("()[] ").split(",")]
    a = np.asarray(x, int).ravel()
    if a.size != 3:
        raise ValueError(f"Bad orientation: {x}")
    return tuple(int(z) for z in a)
def load_catalog_index():
    with open(CATALOG, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        ori = canonical_orientation(r["orientation"])
        site = int(r["site_index"])
        out[(ori, site)] = dict(
            site_id=site,
            orientation=ori,
            kappa=float(r["kappa"]),
            fI_kHz=float(r["fI_Hz"]) / 1e3,
            fm_kHz=float(r["omega_ms_Hz"]) / 1e3,
            fminus_kHz=float(r["f_minus_Hz"]) / 1e3,
            fplus_kHz=float(r["f_plus_Hz"]) / 1e3,
            distance_A=float(r.get("distance_A", np.nan)),
        )
    return out


def stats(y, e, pred, k):
    e = np.maximum(np.asarray(e, float), 1e-12)
    chi2 = float(np.sum(((np.asarray(y, float) - np.asarray(pred, float)) / e) ** 2))
    n = len(y)
    k = int(k)
    red = chi2 / max(1, n - k)
    aic = chi2 + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    bic = chi2 + k * np.log(max(n, 2))
    return dict(chi2=chi2, red_chi2=red, aicc=float(aicc), bic=float(bic), npar=k)
def site_q(t_us, site, dt_us=0.0):
    # Exact single-spin Hahn echo factor used in the previous physical-bath model.
    t = np.asarray(t_us, float) + float(dt_us)
    fI = float(site["fI_kHz"]) / 1000.0
    fm = float(site["fm_kHz"]) / 1000.0
    return 2.0 * float(site["kappa"]) * (
        np.sin(0.5 * np.pi * fI * t) ** 2
    ) * (
        np.sin(0.5 * np.pi * fm * t) ** 2
    )


def product_coherence(t, sites, scales, dt):
    L = np.ones_like(np.asarray(t, float))
    for s, scale in zip(sites, scales):
        L *= 1.0 - float(scale) * site_q(t, s, dt)
    return L


def product_model(t, baseline, contrast, carrier, sites, scales, dt):
    L = product_coherence(t, sites, scales, dt)
    return baseline - contrast * carrier * L


def additive_basis(t, carrier, site, phi0, phi1):
    tt = np.asarray(t, float)
    # Reconstruct the V14 additive control with V14's saved frequencies exactly.
    f0 = float(site["v14_f0_kHz"]) / 1000.0
    f1 = float(site["v14_f1_kHz"]) / 1000.0
    return carrier * (
        np.cos(2*np.pi*f0*tt + phi0) + np.cos(2*np.pi*f1*tt + phi1)
    )
def fit_product(t, y, e, baseline, contrast, carrier, sites, x0_scales,
                free_sites=False, fit_dt=True):
    n = len(sites)
    e = np.maximum(np.asarray(e, float), 1e-12)
    if free_sites:
        x0 = np.r_[np.asarray(x0_scales, float), 0.0] if fit_dt else np.asarray(x0_scales, float)
        lb = np.r_[np.zeros(n), -DT_BOUND_US] if fit_dt else np.zeros(n)
        ub = np.r_[np.full(n, SCALE_MAX), DT_BOUND_US] if fit_dt else np.full(n, SCALE_MAX)
    else:
        seed = float(np.median(x0_scales))
        x0 = np.array([seed, 0.0]) if fit_dt else np.array([seed])
        lb = np.array([0.0, -DT_BOUND_US]) if fit_dt else np.array([0.0])
        ub = np.array([SCALE_MAX, DT_BOUND_US]) if fit_dt else np.array([SCALE_MAX])

    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    def unpack(x):
        if free_sites:
            scales = np.asarray(x[:n], float)
            dt = float(x[n]) if fit_dt else 0.0
        else:
            scales = np.full(n, float(x[0]))
            dt = float(x[1]) if fit_dt else 0.0
        return scales, dt

    def resid(x):
        scales, dt = unpack(x)
        pred = product_model(t, baseline, contrast, carrier, sites, scales, dt)
        return (np.asarray(y, float) - pred) / e
    starts = [x0]
    if fit_dt:
        for dt0 in (-0.20, -0.10, 0.10, 0.20):
            q = x0.copy()
            q[-1] = dt0
            starts.append(np.clip(q, lb + 1e-8, ub - 1e-8))
    fits = []
    for seed in starts:
        try:
            rr = least_squares(
                resid, seed, bounds=(lb, ub), loss="soft_l1", f_scale=1.0,
                max_nfev=20000, x_scale="jac",
            )
            ff = least_squares(
                resid, rr.x, bounds=(lb, ub), loss="linear",
                max_nfev=30000, ftol=1e-11, xtol=1e-11, gtol=1e-11, x_scale="jac",
            )
            scales, dt = unpack(ff.x)
            pred = product_model(t, baseline, contrast, carrier, sites, scales, dt)
            st = stats(y, e, pred, len(ff.x))
            fits.append(dict(scales=scales, dt_us=dt, pred=pred, **st))
        except Exception:
            pass
    if not fits:
        raise RuntimeError("All product-model starts failed")
    return min(fits, key=lambda z: (z["chi2"], z["bic"]))
def analyze_nv(base, t, y, e, row, cat, v15_sites):
    nv = int(row.name) if row.name is not None else int(row["nv_index"])
    order = int(row.model_order)
    ori = canonical_orientation(row.orientation)
    theta = np.asarray(json.loads(row.theta_json), float)
    baseline, contrast, carrier = base.carrier_from_bg(t, theta[:9])
    core = baseline - contrast * carrier

    sites = []
    phases = []
    for j in range(1, order + 1):
        sid = int(row[f"c13_{j}_site_id"])
        site = dict(cat[(ori, sid)])
        site["v14_f0_kHz"] = float(row[f"c13_{j}_f0_kHz"])
        site["v14_f1_kHz"] = float(row[f"c13_{j}_f1_kHz"])
        phases.append((float(row[f"c13_{j}_phi0"]), float(row[f"c13_{j}_phi1"])))
        sites.append(site)

    X = np.column_stack([
        additive_basis(t, carrier, s, ph[0], ph[1])
        for s, ph in zip(sites, phases)
    ])
    amp_scale1 = contrast * np.array([s["kappa"] for s in sites]) / 4.0
    target = np.asarray(y, float) - core
    w = 1.0 / np.maximum(np.asarray(e, float), 1e-12)
    # Additive shared-scale control, with no V14 upper ceiling.
    g = X @ amp_scale1
    denom = float(np.sum((g*w)**2))
    shared_scale = max(0.0, float(np.sum((g*w)*(target*w))) / denom) if denom > 0 else 0.0
    pred_add_shared = core + shared_scale * g
    st_add_shared = stats(y, e, pred_add_shared, 1)

    # Additive independent amplitudes, same as V15.
    free = lsq_linear(X*w[:, None], target*w, bounds=(0.0, np.inf))
    amps = np.asarray(free.x, float)
    free_scales = np.divide(amps, amp_scale1, out=np.zeros_like(amps), where=amp_scale1 > 0)
    pred_add_free = core + X @ amps
    st_add_free = stats(y, e, pred_add_free, order)

    product_shared0 = fit_product(
        t, y, e, baseline, contrast, carrier, sites,
        np.full(order, shared_scale), free_sites=False, fit_dt=False,
    )
    product_shared_dt = fit_product(
        t, y, e, baseline, contrast, carrier, sites,
        np.full(order, shared_scale), free_sites=False, fit_dt=True,
    )
    product_free_dt = fit_product(
        t, y, e, baseline, contrast, carrier, sites,
        np.clip(free_scales, 0.0, SCALE_MAX), free_sites=True, fit_dt=True,
    )
    # Quantify whether fitted product factors remain in the physical coherence range.
    qmax = np.array([np.max(site_q(t, s, product_free_dt["dt_us"])) for s in sites])
    factor_min = 1.0 - product_free_dt["scales"] * qmax

    summary = dict(
        nv_index=nv, model_order=order, site_key=str(row.site_key),
        v14_red_chi2=float(row.red_chi2),
        additive_shared_scale=shared_scale,
        additive_shared_chi2=st_add_shared["chi2"],
        additive_shared_bic=st_add_shared["bic"],
        additive_free_chi2=st_add_free["chi2"],
        additive_free_bic=st_add_free["bic"],
        product_shared0_scale=float(product_shared0["scales"][0]),
        product_shared0_chi2=product_shared0["chi2"],
        product_shared0_bic=product_shared0["bic"],
        product_shared_dt_scale=float(product_shared_dt["scales"][0]),
        product_shared_dt_us=float(product_shared_dt["dt_us"]),
        product_shared_dt_chi2=product_shared_dt["chi2"],
        product_shared_dt_bic=product_shared_dt["bic"],
        product_free_dt_us=float(product_free_dt["dt_us"]),
        product_free_dt_chi2=product_free_dt["chi2"],
        product_free_dt_bic=product_free_dt["bic"],
        delta_bic_product_shared_vs_add_shared=product_shared_dt["bic"]-st_add_shared["bic"],
        delta_bic_product_free_vs_add_free=product_free_dt["bic"]-st_add_free["bic"],
        product_factor_min=float(np.min(factor_min)),
        product_any_factor_negative=bool(np.any(factor_min < 0)),
    )
    site_rows = []
    for j, s in enumerate(sites):
        vv = v15_sites[(v15_sites.nv_index == nv) & (v15_sites.site_id == s["site_id"])]
        site_rows.append(dict(
            nv_index=nv, site_slot=j+1, site_id=s["site_id"],
            kappa=s["kappa"], fI_kHz=s["fI_kHz"], fm_kHz=s["fm_kHz"],
            fminus_kHz=s["fminus_kHz"], fplus_kHz=s["fplus_kHz"],
            additive_free_scale=float(free_scales[j]),
            product_free_scale=float(product_free_dt["scales"][j]),
            product_qmax=float(qmax[j]),
            product_factor_min=float(factor_min[j]),
            v15_r_vs_v14=float(vv.r_vs_v14.iloc[0]) if len(vv) else np.nan,
        ))

    curves = dict(
        add_shared=pred_add_shared,
        add_free=pred_add_free,
        prod_shared0=product_shared0["pred"],
        prod_shared_dt=product_shared_dt["pred"],
        prod_free_dt=product_free_dt["pred"],
    )
    return summary, site_rows, curves


def make_page(t, y, e, summary, site_df, curves):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), height_ratios=[2.2, 1.0])
    ax = axes[0]
    ax.errorbar(t, y, yerr=e, fmt="o", ms=3, capsize=1, label="data")
    ax.plot(t, curves["add_shared"], lw=1.1, label="additive shared")
    ax.plot(t, curves["add_free"], lw=1.2, label="additive free sites")
    ax.plot(t, curves["prod_shared_dt"], lw=1.2, ls="--", label="product shared + dt")
    ax.plot(t, curves["prod_free_dt"], lw=1.5, label="product free sites + dt")
    ax.set(xlabel="Total Hahn-echo evolution time (us)", ylabel="Normalized signal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(
        f"NV {summary['nv_index']} | {summary['site_key']} | "
        f"dBIC product-shared vs additive-shared = "
        f"{summary['delta_bic_product_shared_vs_add_shared']:+.1f}"
    )

    ax = axes[1]
    x = np.arange(len(site_df))
    width = 0.32
    ax.bar(x-width/2, site_df.additive_free_scale, width, label="additive free scale")
    ax.bar(x+width/2, site_df.product_free_scale, width, label="product free scale")
    ax.set_xticks(x, [str(int(v)) for v in site_df.site_id])
    ax.axhline(1.0, ls=":", lw=0.8)
    ax.set(xlabel="Selected C13 site id", ylabel="Effective scale")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def run(outdir, targets):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = v6.load_backend("52G")
    _, checkpoint, _, _ = base.discover_paths()
    t, Y, E = base.load_data(checkpoint)
    cand = pd.read_csv(V14_FILE)
    winners = cand[cand.rank_global_bic == 1].set_index("nv_index")
    v15_sites = pd.read_csv(V15_SITE)
    cat = load_catalog_index()

    summaries, sites_all, curves_all = [], [], {}
    for nv in targets:
        row = winners.loc[int(nv)]
        summary, sr, curves = analyze_nv(
            base, t, Y[int(nv)], E[int(nv)], row, cat, v15_sites
        )
        summaries.append(summary)
        sites_all.extend(sr)
        curves_all[int(nv)] = curves
        print(
            f"NV{nv}: add-free BIC={summary['additive_free_bic']:.2f}, "
            f"prod-shared BIC={summary['product_shared_dt_bic']:.2f}, "
            f"prod-free BIC={summary['product_free_dt_bic']:.2f}"
        )

    sdf = pd.DataFrame(summaries)
    adf = pd.DataFrame(sites_all)
    sdf.to_csv(outdir/"v16_52G_summary.csv", index=False)
    adf.to_csv(outdir/"v16_52G_site_scales.csv", index=False)

    with PdfPages(outdir/"v16_52G_product_diagnostic.pdf") as pdf:
        for _, s in sdf.sort_values("nv_index").iterrows():
            nv = int(s.nv_index)
            fig = make_page(
                t, Y[nv], base.safe_err(E[nv]), s.to_dict(),
                adf[adf.nv_index == nv], curves_all[nv],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    np.savez_compressed(
        outdir/"v16_52G_curves.npz",
        t=np.asarray(t),
        nv_indices=np.asarray(targets, int),
        add_shared=np.vstack([curves_all[n]["add_shared"] for n in targets]),
        add_free=np.vstack([curves_all[n]["add_free"] for n in targets]),
        prod_shared_dt=np.vstack([curves_all[n]["prod_shared_dt"] for n in targets]),
        prod_free_dt=np.vstack([curves_all[n]["prod_free_dt"] for n in targets]),
    )
    print("\nSUMMARY")
    cols = [
        "nv_index","additive_shared_scale","additive_shared_bic","additive_free_bic",
        "product_shared_dt_scale","product_shared_dt_us","product_shared_dt_bic",
        "product_free_dt_bic","delta_bic_product_shared_vs_add_shared",
        "delta_bic_product_free_vs_add_free","product_factor_min",
        "product_any_factor_negative",
    ]
    print(sdf[cols].to_string(index=False))
    print("\nSaved:", outdir)
    return sdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nv", default=",".join(str(x) for x in TARGET_NVS))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    targets = [int(x) for x in args.nv.split(",") if x.strip()]
    run(args.output_dir, targets)


if __name__ == "__main__":
    main()
