"""V24 targeted heavy rerank for the QNami 52 G V22 problem NVs.

Reads the completed baseline V22 QC table, selects only NVs with serious flags:
  high_redchi2, high_scale_gt10, scale_bound, line_amp_gt2contrast,
  relative_kappa_mismatch

Then reruns V22 on that subset with a heavier search:
  local pool 12, up to two local substitutions, more parent candidates,
  larger augmentation cap, and deeper polish.

This intentionally does NOT use the unrelated Johnson 49 G dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import sc_c13_c13_spin_echo_v22_full_physical_rerank as v22

BASE=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\c13_spin_echo_v22_full_physical_rerank\2026_09"
    r"\52G\smax30_tol12_pool8_sub1_topO2_topG4_cap40"
)
DEFAULT_OUT=Path(
    r"G:\nvdata\pc_NVOffice\branch_master\c13_spin_echo\spin_echo_v24_targeted_heavy_52G\2026_09"
)
SERIOUS_FLAGS={
    "high_redchi2",
    "high_scale_gt10",
    "scale_bound",
    "line_amp_gt2contrast",
    "relative_kappa_mismatch",
}


def select_nvs(qc_path):
    d=pd.read_csv(qc_path).fillna({"qc_flags":""})
    keep=[]
    reasons={}
    for r in d.itertuples():
        flags={x for x in str(r.qc_flags).split(";") if x}
        hit=sorted(flags & SERIOUS_FLAGS)
        if hit:
            nv=int(r.nv_index)
            keep.append(nv)
            reasons[nv]=";".join(hit)
    return keep,reasons


def run(args):
    qc_path=BASE/"v22_qc_summary.csv"
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    nvs,reasons=select_nvs(qc_path)
    if not nvs:
        print("No serious-QC NVs selected; nothing to run.")
        return

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(
        [{"nv_index":nv,"selection_reason":reasons[nv]} for nv in nvs]
    ).to_csv(out/"v24_selected_nvs.csv",index=False)

    print(f"V24 selected {len(nvs)} NVs:")
    print(",".join(map(str,nvs)))
    cfg=SimpleNamespace(
        field="52G",
        workers=int(args.workers),
        scale_max=30.0,
        stage1_max_nfev=2000,
        polish_per_order=5,
        polish_delta_bic=12.0,
        polish_cap_per_order=12,
        freq_tol_khz=12.0,
        local_pool_size=12,
        max_local_substitutions=2,
        max_augmented_per_base=120,
        augment_top_per_order=4,
        augment_top_global=8,
        augment_stage1_max_nfev=3000,
        save_top_n=15,
        nv=",".join(map(str,nvs)),
        output_dir=str(out),
    )
    final,qc=v22.run(cfg)
    print("\nV24 TARGETED HEAVY COMPLETE")
    print("NVs:",len(qc))
    print("winner orders:",qc.model_order.value_counts().sort_index().to_dict())
    print("QC still flagged:",int((qc.qc_flags.fillna("")!="").sum()))
    print("output:",out)
    return final,qc


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--output-dir",default=str(DEFAULT_OUT))
    args=ap.parse_args()
    run(args)


if __name__=="__main__":
    main()
