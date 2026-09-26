"""V17 efficient bound-release rerank of the full V14 candidate table.

Candidates whose V14 shared visibility optimum is safely below the old cap are
unchanged when s_max moves 3 -> 5.  Refit every near-bound candidate, then
robustly polish all selection-relevant candidates and rerank globally.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import sc_spin_echo_physical_family_search_v6 as v6
import sc_spin_echo_v14_beta2_taper0_rerank as v14
import sc_spin_echo_v15_free_site_amplitude_diagnostic as v15

v6.VISIBILITY_SCALE_MAX = 5.0
MODEL_TAG = "v17_beta2_taper0_smax5_boundrelease"
RELEASE_THRESHOLD = 2.90
DEFAULT_OUT = Path(
    r"G:\nvdata\pc_NVOffice\branch_master"
    r"\spin_echo_v17_beta2_taper0_smax5_boundrelease\2026_09"
)
def stage1_release_nv(field, nv, records, t, y, e, checkpoint, max_nfev):
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        return str(checkpoint)
    base = v6.load_backend(field)
    out = []
    with threadpool_limits(limits=1):
        for rec in records:
            sites = v14.sites_from_record(rec)
            fit = v14.projected_seed_fit(
                base, t, y, e, rec, sites, max_nfev
            )
            out.append(v14.fit_to_updates(
                base, t, rec, sites, fit, "bound_release"
            ))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(checkpoint, index=False, compression="gzip")
    return str(checkpoint)


def winner_table(df, prefix):
    w = df[df.rank_global_bic == 1].copy()
    cols = [
        "nv_index", "model_order", "site_key", "bic", "red_chi2",
        "visibility_scale", "visibility_scale_bound_hit",
    ]
    w = w[cols].copy()
    w.columns = ["nv_index"] + [f"{prefix}_{c}" for c in cols[1:]]
    return w
def run(args):
    t0 = time.time()
    field = "52G"
    src_path = v15.find_v14_file(field)
    src = pd.read_csv(src_path)
    src["_row_id"] = np.arange(len(src), dtype=int)

    base = v6.load_backend(field)
    _, checkpoint, _, _ = base.discover_paths()
    t, Y, E = base.load_data(checkpoint)

    release_mask = (
        (src.model_order > 0)
        & (src.visibility_scale.fillna(-1.0) >= float(args.release_threshold))
    )
    release = src[release_mask].copy()
    nvs = sorted(release.nv_index.unique().astype(int).tolist())

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ck1 = outdir / "checkpoint_release_52G"
    ck2 = outdir / "checkpoint_polish_52G"

    compact = release[v14.compact_columns(release)].copy()
    records = {
        nv: compact[compact.nv_index == nv].to_dict("records")
        for nv in nvs
    }
    jobs = []
    for nv in nvs:
        cp = ck1 / f"nv_{nv:04d}.csv.gz"
        if not cp.exists():
            jobs.append((nv, records[nv], cp))
    print(
        f"V17 release: {len(release):,}/{len(src):,} candidates "
        f"across {len(nvs)} NVs; {len(jobs)} checkpoints remaining"
    )
    if jobs:
        Parallel(
            n_jobs=max(1, args.workers), backend="loky", verbose=10
        )(
            delayed(stage1_release_nv)(
                field, nv, recs, t, Y[nv], E[nv], str(cp), args.max_nfev
            )
            for nv, recs, cp in jobs
        )

    updates = pd.concat(
        [pd.read_csv(ck1 / f"nv_{nv:04d}.csv.gz") for nv in nvs],
        ignore_index=True,
    )
    work = v14.apply_updates(src, updates)
    work["visibility_scale_max"] = np.where(
        work.model_order > 0, 5.0, np.nan
    )
    work["amplitude_model"] = "shared_scale_times_contrast_kappa_over_4_smax5"
    work["v17_release_threshold"] = float(args.release_threshold)

    chosen = v14.choose_polish_rows(
        work, args.polish_per_order,
        args.polish_delta_bic, args.polish_cap_per_order,
    )
    polish = work[work._row_id.isin(chosen)].copy()
    print(f"V17 polish: {len(polish):,}/{len(work):,} candidates")
    fitneed = [
        "_row_id", "nv_index", "model_order", "theta_json", "baseline",
        "orientation", "site_key", "v6_source_bic",
    ]
    for j in (1, 2, 3):
        for c in ["site_id", "f0_kHz", "f1_kHz", "kappa"]:
            fitneed.append(f"c13_{j}_{c}")
    pcompact = polish[[c for c in fitneed if c in polish.columns]].copy()
    pcompact["stage1_theta_json"] = pcompact["theta_json"]

    precs = {
        int(nv): pcompact[pcompact.nv_index == nv].to_dict("records")
        for nv in sorted(pcompact.nv_index.unique())
    }
    jobs = []
    for nv, recs in precs.items():
        cp = ck2 / f"nv_{nv:04d}.csv.gz"
        if not cp.exists():
            jobs.append((nv, recs, cp))
    if jobs:
        Parallel(
            n_jobs=max(1, args.workers), backend="loky", verbose=10
        )(
            delayed(v14.stage2_nv)(
                field, nv, recs, t, Y[nv], E[nv], str(cp)
            )
            for nv, recs, cp in jobs
        )
    stage2_files = [ck2 / f"nv_{nv:04d}.csv.gz" for nv in precs]
    stage2 = pd.concat(
        [pd.read_csv(p) for p in stage2_files if p.exists()],
        ignore_index=True,
    )
    if len(stage2):
        work = v14.apply_updates(work, stage2)

    final = v14.rerank(work)
    tag = f"{Path(src_path).stem.replace('_candidate_fits','')}_{MODEL_TAG}"
    outcsv = outdir / f"{tag}_candidate_fits.csv.gz"
    final.to_csv(outcsv, index=False, compression="gzip")

    oldw = winner_table(src, "v14")
    neww = winner_table(final, "v17")
    trans = oldw.merge(neww, on="nv_index", how="outer")
    trans["order_changed"] = trans.v14_model_order != trans.v17_model_order
    trans["site_changed"] = (
        trans.v14_site_key.astype(str) != trans.v17_site_key.astype(str)
    )
    trans["delta_bic_v17_minus_v14"] = trans.v17_bic - trans.v14_bic
    trans.to_csv(outdir / f"{tag}_winner_transitions.csv", index=False)

    counts = (
        neww.v17_model_order.value_counts().reindex([0,1,2,3], fill_value=0)
        .rename_axis("model_order").reset_index(name="count")
    )
    counts.to_csv(outdir / f"{tag}_winner_order_counts.csv", index=False)
    selected = final[final.rank_global_bic == 1].copy()
    target = trans[trans.nv_index.isin([171,37,168,85])].copy()
    meta = dict(
        field=field,
        source_v14=str(src_path),
        output_candidate=str(outcsv),
        release_threshold=float(args.release_threshold),
        visibility_scale_max=5.0,
        n_candidates=int(len(final)),
        n_release_candidates=int(len(release)),
        n_polished=int(len(stage2)),
        n_nvs=int(final.nv_index.nunique()),
        order_changes=int(trans.order_changed.sum()),
        site_changes=int(trans.site_changed.sum()),
        winner_bound_hits_at5=int(
            selected.visibility_scale_bound_hit.fillna(False).sum()
        ),
        median_winner_scale=float(
            selected.loc[selected.model_order > 0, "visibility_scale"].median()
        ),
        elapsed_minutes=float((time.time() - t0) / 60.0),
    )
    with open(outdir / f"{tag}_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nV17 COMPLETE")
    print(json.dumps(meta, indent=2))
    print("\nWinner order counts:")
    print(counts.to_string(index=False))
    print("\nTarget four transitions:")
    show = [
        "nv_index", "v14_model_order", "v17_model_order",
        "v14_site_key", "v17_site_key",
        "v14_visibility_scale", "v17_visibility_scale",
        "v14_red_chi2", "v17_red_chi2",
        "delta_bic_v17_minus_v14",
    ]
    print(target[show].to_string(index=False))
    print("\nLargest BIC improvements:")
    print(
        trans.nsmallest(20, "delta_bic_v17_minus_v14")[
            ["nv_index", "v14_model_order", "v17_model_order",
             "v14_visibility_scale", "v17_visibility_scale",
             "delta_bic_v17_minus_v14"]
        ].to_string(index=False)
    )
    print("\nOutput:", outcsv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-nfev", type=int, default=800)
    ap.add_argument("--release-threshold", type=float, default=RELEASE_THRESHOLD)
    ap.add_argument("--polish-per-order", type=int, default=3)
    ap.add_argument("--polish-delta-bic", type=float, default=12.0)
    ap.add_argument("--polish-cap-per-order", type=int, default=8)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
