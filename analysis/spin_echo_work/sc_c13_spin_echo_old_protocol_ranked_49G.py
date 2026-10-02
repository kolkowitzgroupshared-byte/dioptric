# -*- coding: utf-8 -*-
"""
Old-protocol, multi-start, ranked spin-echo fitter for the older 49 G Johnson data.

Uses the existing spin_echo_work fine_decay model and optimizer helpers.
Fits the already-combined normalized dataset, searches physical ESEEM catalog
pairs, tries multiple amplitude/phase/revival starts and optimizers, refines
several local minima, and saves ALL attempts plus ranked alternatives.

Time axis is total Hahn-echo evolution time (2*tau), matching the old fitter.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits

from utils import data_manager as dm
from utils import kplotlib as kpl
from analysis.spin_echo_work import fitter_module_for_spin_echo as oldfit
fine_decay = oldfit.fine_decay
from analysis.spin_echo_work.kappa_modulation_depth import build_essem_catalog_with_kappa

# ---------------- user settings ----------------
FILE_STEM = "2025_11_15-14_11_49-johnson_204nv_s9-17d44b"
B_VECTOR_G = np.array([-46.19581364, -17.44900422, -5.57935388], float)
# Reconstructed from the saved 49 G exact-kappa catalog metadata:
# |B| = 49.6955746 G, B_hat = [-0.9295760, -0.3511179, -0.1122706].
GAMMA_C13_KHZ_PER_G = 1.0705
CATALOG_JSON = Path(r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_49G.json")
CATALOG_CSV = Path(r"analysis\spin_echo_work\essem_freq_kappa_catalog_22A_49G.csv")
HYPERFINE_PATH = Path(r"analysis\nv_hyperfine_coupling\nv-2.txt")
NV_INDICES = None

MAX_CATALOG_PAIRS_PER_NV = 1500
SCREEN_MAX_NFEV = 1200
SCREEN_KEEP = 16
AMP_WINDOWS = ((-0.6, 0.6), (-1.0, 1.0), (-2.0, 2.0))
PHASE_STARTS = ((0.0, 0.0), (np.pi / 2, 0.0), (0.0, np.pi / 2), (np.pi / 2, np.pi / 2))
REVIVAL_SEED_RELATIVE = (0.92, 1.00, 1.08)
HEAVY_LSQ_MAX_NFEV = 30000
HEAVY_CURVE_FIT_MAXFEV = 40000
REFINE_MAX_NFEV = 100000
REFINE_TOP_N = 6
TOP_N_SAVE = 10
TOP_N_PLOT = 3
FREQ_LOCK_EPS = 1e-6

CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, CPU_COUNT - 2)
BLAS_THREADS_PER_WORKER = 1

SAVE_ALL_ATTEMPTS = True
SAVE_PDFS = True
SHOW_SUMMARY = True
PDF_COLS, PDF_ROWS = 3, 4
OUTPUT_BASENAME = "spin_echo_old_protocol_ranked_49G"

B_MAG_G = float(np.linalg.norm(B_VECTOR_G))
C13_LARMOR_KHZ = GAMMA_C13_KHZ_PER_G * B_MAG_G
REVIVAL_TOTAL_US_THEORY = 2000.0 / C13_LARMOR_KHZ


def safe_err(x, floor=1e-3):
    x = np.abs(np.asarray(x, float))
    good = np.isfinite(x) & (x > 0)
    fallback = float(np.nanmedian(x[good])) if np.any(good) else floor
    return np.maximum(np.where(good, x, fallback), floor)


def infer_band(t):
    t = np.unique(np.asarray(t, float))
    span = max(float(np.ptp(t)), 1e-9)
    dt = np.diff(t)
    dt = dt[dt > 0]
    dt_min = float(np.min(dt)) if dt.size else span
    return max(0.001, 0.5 / span), 0.49 / dt_min


def calc_stats(y, e, yfit, npar):
    e = safe_err(e)
    chi2 = float(np.sum(((y - yfit) / e) ** 2))
    n, k = len(y), int(npar)
    red = chi2 / max(1, n - k)
    aic = chi2 + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n > k + 1 else np.inf
    return chi2, red, float(aicc)


def get_output_base():
    p = Path(dm.get_file_path(__file__, dm.get_time_stamp(), OUTPUT_BASENAME)).with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_data():
    data = dm.get_raw_data(file_stem=FILE_STEM, load_npz=True)
    nv_list = data["nv_list"]
    y = np.asarray(data["norm_counts"], float)
    e = safe_err(data["norm_counts_ste"])
    if "total_evolution_times" in data:
        t = np.asarray(data["total_evolution_times"], float).ravel()
    else:
        t = 2.0 * np.asarray(data["taus"], float).ravel() / 1e3
    order = np.argsort(t)
    t, y, e = t[order], y[:, order], e[:, order]
    ori = None
    if "orientations" in data:
        arr = np.asarray(data["orientations"], int)
        if arr.ndim == 2 and arr.shape[0] >= len(nv_list) and arr.shape[1] == 3:
            ori = arr[: len(nv_list)]
    return data, nv_list, t, y, e, ori


def load_catalog():
    if not CATALOG_JSON.exists():
        if not HYPERFINE_PATH.exists():
            raise FileNotFoundError(HYPERFINE_PATH)
        CATALOG_JSON.parent.mkdir(parents=True, exist_ok=True)
        build_essem_catalog_with_kappa(
            hyperfine_path=str(HYPERFINE_PATH),
            B_lab_vec=B_VECTOR_G * 1e-4,
            orientations=((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)),
            distance_max_A=22.0,
            gamma_n_Hz_per_T=GAMMA_C13_KHZ_PER_G * 1e7,
            p_occ=0.011,
            ms=-1,
            phi_deg=0.0,
            out_json=str(CATALOG_JSON),
            out_csv=str(CATALOG_CSV),
            read_hf_table_fn=None,
        )
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pairs(records, orientation, band):
    lo, hi = band
    ori = None
    if orientation is not None:
        a = np.asarray(orientation, int).ravel()
        if a.size == 3 and np.any(a):
            ori = tuple(int(v) for v in a)
    out = []
    for r in records:
        ro = tuple(int(v) for v in r.get("orientation", ()))
        if ori is not None and ro != ori:
            continue
        try:
            fp = float(r["f_plus_Hz"]) / 1e6
            fm = float(r["f_minus_Hz"]) / 1e6
        except Exception:
            continue
        f0, f1 = max(fp, fm), min(fp, fm)
        if lo <= f0 <= hi and lo <= f1 <= hi:
            out.append(
                dict(
                    f0=f0, f1=f1, site_id=int(r.get("site_index", -1)),
                    orientation=ro, kappa=float(r.get("kappa", 0.0)),
                    distance_A=float(r.get("distance_A", np.nan)),
                )
            )
    out.sort(key=lambda c: (-np.nan_to_num(c["kappa"]), c["site_id"]))
    uniq, seen = [], set()
    for c in out:
        key = (c["orientation"], round(c["f0"], 7), round(c["f1"], 7))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq if MAX_CATALOG_PAIRS_PER_NV is None else uniq[:MAX_CATALOG_PAIRS_PER_NV]


def base_vectors(t, y):
    p0, lb, ub = oldfit._initial_guess_and_bounds(t, y, True, fixed_rev_time=None)
    pmap = oldfit._param_index_map(fine_decay)
    p0[pmap["revival_time"]] = np.clip(
        REVIVAL_TOTAL_US_THEORY,
        lb[pmap["revival_time"]] + 1e-6,
        ub[pmap["revival_time"]] - 1e-6,
    )
    return p0, lb, ub, pmap


def seeded_vectors(t, y, cand, amp, phase, rev_seed, warm=None):
    p0, lb, ub, pmap = base_vectors(t, y)
    oldfit._set_osc_amp_bounds(lb, ub, fine_decay, *amp)
    p0[pmap["revival_time"]] = np.clip(rev_seed, lb[pmap["revival_time"]] + 1e-7, ub[pmap["revival_time"]] - 1e-7)
    p0[pmap["osc_f0"]], p0[pmap["osc_f1"]] = cand["f0"], cand["f1"]
    for name, val in (("osc_f0", cand["f0"]), ("osc_f1", cand["f1"])):
        i = pmap[name]
        lb[i], ub[i] = val - FREQ_LOCK_EPS, val + FREQ_LOCK_EPS
    p0[pmap["osc_phi0"]], p0[pmap["osc_phi1"]] = phase
    p0, lb, ub = oldfit._retie_contrast_to_baseline(p0, lb, ub, pmap, eps=0.01)
    if warm is not None and np.asarray(warm).shape == p0.shape:
        p0 = np.asarray(warm, float).copy()
        p0[pmap["osc_f0"]], p0[pmap["osc_f1"]] = cand["f0"], cand["f1"]
    eps = 1e-9 * np.maximum(1.0, ub - lb)
    p0 = np.minimum(np.maximum(p0, lb + eps), ub - eps)
    return p0, lb, ub, pmap


def public(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def fit_attempt(nv, t, y, e, cand, amp, phase, rev, stage, mode, budget, warm=None):
    p0, lb, ub, pmap = seeded_vectors(t, y, cand, amp, phase, rev, warm)
    base = dict(
        nv_index=int(nv), stage=stage, mode=mode, status="fail",
        site_id=int(cand["site_id"]), orientation=str(tuple(cand["orientation"])),
        kappa=float(cand["kappa"]), distance_A=float(cand["distance_A"]),
        f0_kHz=1000 * cand["f0"], f1_kHz=1000 * cand["f1"],
        amp_min=float(amp[0]), amp_max=float(amp[1]),
        phase0_seed=float(phase[0]), phase1_seed=float(phase[1]),
        revival_seed_us=float(rev), red_chi2=np.inf, aicc=np.inf,
        score_primary=np.inf, score_amp_tie=np.inf, popt_json="", error="",
    )
    try:
        if mode == "least_squares":
            popt, _, _ = oldfit._fit_least_squares(fine_decay, t, y, e, p0, lb, ub, max_nfev=int(budget))
        elif mode == "curve_fit":
            popt, _, _ = oldfit._fit_curve_fit(fine_decay, t, y, e, p0, lb, ub, maxfev=int(budget))
        else:
            raise ValueError(mode)
        curve = fine_decay(t, *popt)
        _, red, aicc = calc_stats(y, e, curve, len(popt))
        score = oldfit._score_tuple(popt, red, lb, ub, pmap)
        base.update(
            status="ok", red_chi2=float(red), aicc=float(aicc),
            score_primary=float(score[0]), score_amp_tie=float(score[1]),
            popt_json=json.dumps(np.asarray(popt, float).tolist()),
        )
        base["_popt"] = np.asarray(popt, float)
        base["_curve"] = np.asarray(curve, float)
    except Exception as exc:
        base["error"] = str(exc)
    return base


def same_solution(a, b):
    if a["site_id"] != b["site_id"] or "_popt" not in a or "_popt" not in b:
        return False
    pa, pb = a["_popt"], b["_popt"]
    scale = np.maximum(1e-4, np.abs(pa) + np.abs(pb))
    return bool(np.max(np.abs(pa - pb) / scale) < 5e-3)


def rank_distinct(attempts, top_n=TOP_N_SAVE):
    good = [a for a in attempts if a["status"] == "ok" and "_popt" in a and np.isfinite(a["score_primary"])]
    good.sort(key=lambda a: (a["score_primary"], a["score_amp_tie"], a["red_chi2"], a["aicc"]))
    out = []
    for c in good:
        if not any(same_solution(c, x) for x in out):
            out.append(c)
        if len(out) >= top_n:
            break
    for i, c in enumerate(out, 1):
        c["rank"] = i
    return out


def fit_one_nv(nv, t, y, e, orientation, catalog, band):
    attempts = []
    pairs = get_pairs(catalog, orientation, band)
    if not pairs:
        return dict(nv_index=int(nv), status="no_pairs", attempts=[], top=[])

    # broad cheap screen, but with the actual ESEEM pair included
    for c in pairs:
        attempts.append(
            fit_attempt(
                nv, t, y, e, c, (-1.0, 1.0), (0.0, 0.0),
                REVIVAL_TOTAL_US_THEORY, "screen", "least_squares", SCREEN_MAX_NFEV,
            )
        )

    screen = [a for a in attempts if a["status"] == "ok"]
    screen.sort(key=lambda a: (a["score_primary"], a["red_chi2"]))
    survivors, seen = [], set()
    for a in screen:
        key = (a["orientation"], a["site_id"], round(a["f0_kHz"], 3), round(a["f1_kHz"], 3))
        if key not in seen:
            seen.add(key)
            survivors.append(a)
        if len(survivors) >= SCREEN_KEEP:
            break
    if not survivors:
        return dict(nv_index=int(nv), status="screen_failed", attempts=attempts, top=[])

    rev_seeds = [round(REVIVAL_TOTAL_US_THEORY * r, 6) for r in REVIVAL_SEED_RELATIVE] + [28.0]
    for s in survivors:
        c = dict(
            f0=s["f0_kHz"] / 1000.0, f1=s["f1_kHz"] / 1000.0,
            site_id=s["site_id"], orientation=eval(s["orientation"]),
            kappa=s["kappa"], distance_A=s["distance_A"],
        )
        for amp in AMP_WINDOWS:
            for phase in PHASE_STARTS:
                for rev in rev_seeds:
                    attempts.append(fit_attempt(nv, t, y, e, c, amp, phase, rev, "multistart", "least_squares", HEAVY_LSQ_MAX_NFEV))
                    attempts.append(fit_attempt(nv, t, y, e, c, amp, phase, rev, "multistart", "curve_fit", HEAVY_CURVE_FIT_MAXFEV))

    # refine several distinct hypotheses, not only the current winner
    preliminary = rank_distinct(attempts, max(REFINE_TOP_N, TOP_N_SAVE))
    for a in preliminary[:REFINE_TOP_N]:
        c = dict(
            f0=a["f0_kHz"] / 1000.0, f1=a["f1_kHz"] / 1000.0,
            site_id=a["site_id"], orientation=eval(a["orientation"]),
            kappa=a["kappa"], distance_A=a["distance_A"],
        )
        attempts.append(
            fit_attempt(
                nv, t, y, e, c, (a["amp_min"], a["amp_max"]),
                (a["phase0_seed"], a["phase1_seed"]), a["revival_seed_us"],
                "refine", "least_squares", REFINE_MAX_NFEV, warm=a["_popt"],
            )
        )

    top = rank_distinct(attempts, TOP_N_SAVE)
    return dict(nv_index=int(nv), status=("ok" if top else "failed"), attempts=attempts, top=top)


def resolve_nv_indices(n):
    if NV_INDICES is None:
        return np.arange(n, dtype=int)
    x = np.asarray(NV_INDICES, int)
    return np.unique(x[(x >= 0) & (x < n)])


def run_all(nv_list, t, y, e, orientations, catalog):
    inds = resolve_nv_indices(len(nv_list))
    band = infer_band(t)
    print(f"|B|={B_MAG_G:.4f} G, expected 2tau revival={REVIVAL_TOTAL_US_THEORY:.4f} us")
    print(f"band={1000*band[0]:.1f}-{1000*band[1]:.1f} kHz, workers={N_JOBS}")
    print(f"catalog cap={MAX_CATALOG_PAIRS_PER_NV}, survivors={SCREEN_KEEP}")

    def task(i):
        ori = orientations[i] if orientations is not None else None
        try:
            r = fit_one_nv(int(i), t, y[i], e[i], ori, catalog, band)
            if r["top"]:
                b = r["top"][0]
                print(f"[NV {i:3d}] redchi={b['red_chi2']:.3g}, site={b['site_id']}, attempts={len(r['attempts'])}")
            return r
        except Exception as exc:
            return dict(nv_index=int(i), status="exception", error=str(exc), traceback=traceback.format_exc(), attempts=[], top=[])

    with threadpool_limits(limits=BLAS_THREADS_PER_WORKER):
        results = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=1, verbose=5)(delayed(task)(int(i)) for i in inds)
    return results, band


def flatten(results):
    all_rows, top_rows, best_rows = [], [], []
    for r in results:
        all_rows += [public(a) for a in r["attempts"]]
        for rank, a in enumerate(r["top"], 1):
            row = public(a)
            row["rank"] = rank
            top_rows.append(row)
            if rank == 1:
                best_rows.append(row)
    return pd.DataFrame(all_rows), pd.DataFrame(top_rows), pd.DataFrame(best_rows)


def save_checkpoint(base, t, y, e, results):
    n_nv, n_t = y.shape
    curves = np.full((n_nv, TOP_N_SAVE, n_t), np.nan)
    params = np.full((n_nv, TOP_N_SAVE, 14), np.nan)
    red = np.full((n_nv, TOP_N_SAVE), np.nan)
    score = np.full((n_nv, TOP_N_SAVE), np.nan)
    site = np.full((n_nv, TOP_N_SAVE), -1, int)
    for r in results:
        i = r["nv_index"]
        for j, a in enumerate(r["top"][:TOP_N_SAVE]):
            curves[i, j] = a["_curve"]
            p = a["_popt"]
            params[i, j, : min(14, len(p))] = p[:14]
            red[i, j], score[i, j], site[i, j] = a["red_chi2"], a["score_primary"], a["site_id"]
    path = Path(str(base) + "_fit_checkpoint.npz")
    np.savez_compressed(
        path, source_file_stem=np.asarray([FILE_STEM]), B_vector_G=B_VECTOR_G,
        expected_revival_total_us=np.asarray([REVIVAL_TOTAL_US_THEORY]),
        times_us=t, norm_counts=y, norm_counts_ste=e,
        top_curves=curves, top_params=params, top_red_chi2=red,
        top_score=score, top_site_id=site,
    )
    print(f"Saved: {path}")


def save_pdf(path, t, y, e, results, top_n=1, zoom=False):
    npp = PDF_COLS * PDF_ROWS
    with PdfPages(path) as pdf:
        for start in range(0, len(results), npp):
            page = results[start : start + npp]
            fig, axes = plt.subplots(PDF_ROWS, PDF_COLS, figsize=(15, 13.6), squeeze=False)
            axes = axes.ravel()
            for ax, r in zip(axes, page):
                i = r["nv_index"]
                m = np.abs(t - REVIVAL_TOTAL_US_THEORY) <= 12.5 if zoom else np.ones(len(t), bool)
                ax.errorbar(t[m], y[i, m], yerr=e[i, m], fmt="o", ms=2.5, capsize=1, lw=0.6, label="data")
                for rank, a in enumerate(r["top"][:top_n], 1):
                    ls = "-" if rank == 1 else ("--" if rank == 2 else ":")
                    ax.plot(t[m], a["_curve"][m], ls=ls, lw=1.3 if rank == 1 else 1.0,
                            label=f"#{rank} chi2r={a['red_chi2']:.2f} site={a['site_id']}")
                ax.axvline(REVIVAL_TOTAL_US_THEORY, ls=":", lw=0.7)
                ax.set_title(f"NV {i}", fontsize=9)
                ax.set_xlabel("Total evolution (us)", fontsize=8)
                ax.set_ylabel("Norm. signal", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=5.5)
            for ax in axes[len(page):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved: {path}")


def make_summary(t, y, results):
    good = [r for r in results if r["top"]]
    inds = np.asarray([r["nv_index"] for r in good], int)
    best = np.asarray([r["top"][0]["red_chi2"] for r in good], float)
    second = np.asarray([r["top"][1]["red_chi2"] if len(r["top"]) > 1 else np.nan for r in good], float)
    dscore = np.asarray([
        r["top"][1]["score_primary"] - r["top"][0]["score_primary"] if len(r["top"]) > 1 else np.nan
        for r in good
    ], float)
    fitmed = np.nanmedian(np.vstack([r["top"][0]["_curve"] for r in good]), axis=0)
    datamed = np.nanmedian(y[inds], axis=0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].plot(t, datamed, "o", ms=3, label="median data")
    ax[0, 0].plot(t, fitmed, lw=1.6, label="median best fit")
    ax[0, 0].axvline(REVIVAL_TOTAL_US_THEORY, ls="--", lw=1, label="49 G revival")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.2); ax[0, 0].set_xlabel("Total evolution (us)")
    ax[0, 1].hist(best[np.isfinite(best)], bins=30); ax[0, 1].set_title("Best reduced chi-square")
    m = np.isfinite(best) & np.isfinite(second)
    ax[1, 0].scatter(best[m], second[m], s=14); ax[1, 0].set_xlabel("rank 1 chi2r"); ax[1, 0].set_ylabel("rank 2 chi2r")
    ax[1, 1].hist(dscore[np.isfinite(dscore)], bins=30); ax[1, 1].set_title("rank-2 minus rank-1 score")
    for a in ax.ravel(): a.grid(alpha=0.2)
    fig.suptitle(f"Old-protocol multi-start ranked fits | {len(good)} successful NVs")
    fig.tight_layout()
    return fig


def main():
    kpl.init_kplotlib()
    _, nv_list, t, y, e, orientations = load_data()
    catalog = load_catalog()
    results, band = run_all(nv_list, t, y, e, orientations, catalog)
    base = get_output_base()
    all_df, top_df, best_df = flatten(results)

    # expensive work saved before plotting
    save_checkpoint(base, t, y, e, results)
    if SAVE_ALL_ATTEMPTS:
        p = Path(str(base) + "_all_attempts.csv.gz")
        all_df.to_csv(p, index=False, compression="gzip")
        print(f"Saved: {p}")
    p = Path(str(base) + "_top_fits.csv"); top_df.to_csv(p, index=False); print(f"Saved: {p}")
    p = Path(str(base) + "_best_fits.csv"); best_df.to_csv(p, index=False); print(f"Saved: {p}")

    print(f"Successful NVs: {sum(bool(r['top']) for r in results)}/{len(results)}")
    print(f"Recorded attempts: {len(all_df)}")

    fig = make_summary(t, y, results)
    fig.savefig(Path(str(base) + "_summary.png"), dpi=300, bbox_inches="tight")
    fig.savefig(Path(str(base) + "_summary.pdf"), bbox_inches="tight")
    if SAVE_PDFS:
        save_pdf(Path(str(base) + "_best_fits.pdf"), t, y, e, results, top_n=1, zoom=False)
        save_pdf(Path(str(base) + "_top3_fits.pdf"), t, y, e, results, top_n=TOP_N_PLOT, zoom=False)
        save_pdf(Path(str(base) + "_top3_first_revival.pdf"), t, y, e, results, top_n=TOP_N_PLOT, zoom=True)
    if SHOW_SUMMARY:
        plt.show(block=True)
    else:
        plt.close(fig)
    return results, all_df, top_df, best_df


if __name__ == "__main__":
    main()
