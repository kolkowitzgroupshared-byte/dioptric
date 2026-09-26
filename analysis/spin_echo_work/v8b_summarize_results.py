"""Write a compact, reproducible V8b analysis report for the 212-NV run."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
def pct(x):return f"{100*x:.1f}%"
def table(df,cols,n=10):
    return df[cols].head(n).round(3).to_markdown(index=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--summary",required=True)
    args=ap.parse_args();path=Path(args.summary);base=str(path).removesuffix("_nv_summary.csv")
    s=pd.read_csv(path).sort_values("nv_index")
    c=pd.read_csv(base+"_candidate_fits.csv")
    o=pd.read_csv(base+"_model_orders.csv")
    cfg=json.load(open(base+"_config.json"))
    v6=pd.read_csv(cfg["v6_seed_csv"])
    v6=v6[v6.site_rank==1][["nv_index","red_chi2"]].rename(columns={"red_chi2":"v6_red"})
    joined=s.merge(v6,on="nv_index",validate="one_to_one")
    v7files=list(path.parent.glob("*v7_physical_nv_summary.csv"))
    v7=pd.read_csv(v7files[0]) if v7files else None
    selected=s[s.selected_order>0].copy()
    good=selected[(selected.model_adequacy=="adequate")&
                  (selected.orientation_quality_flag=="ok")]
    supported=good[good.site_inference_status=="conditional_support"]
    large=s[s.selected_red_chi2>3]
    upper=s[s.T2_upper_hit]
    eta=selected.eta_boundary_hit.fillna(False)
    cutoff=s.T2_inference_status.value_counts().to_dict()
    alt=s.orientation_quality_flag.value_counts().to_dict()
    orders=s.selected_order.value_counts().sort_index().to_dict()
    sparse_alias_nv=selected[(selected.c13_1_fplus_kHz/2>500/2.4)|
                             (selected.c13_2_fplus_kHz/2>500/2.4)]
    cvfile=path.parent/"v8b_heldout_validation_late.csv"
    family40=Path(base+"_site_families_40kHz.csv")
    family_sensitivity=""
    if family40.exists():
        fam=pd.read_csv(family40)
        top=fam[fam.family_rank==1].groupby("model_order").family_weight_within_shortlist.median()
        family_sensitivity=(f"- With 40-kHz frequency bins, the median leading family weight is "
                            f"{top.get(1,np.nan):.3f} for singles and {top.get(2,np.nan):.3f} for pairs (within searched shortlists).")
    search_sensitivity=""
    if cfg.get("quick_source") and cfg.get("deep_source"):
        deep=pd.read_csv(cfg["deep_source"]+"_nv_summary.csv")
        quick=pd.read_csv(cfg["quick_source"]+"_nv_summary.csv")
        paired=deep.merge(quick,on="nv_index",suffixes=("_deep","_quick"),validate="one_to_one")
        changed=(paired.selected_site_key_deep!=paired.selected_site_key_quick).sum()
        changed_order=(paired.selected_order_deep!=paired.selected_order_quick).sum()
        search_sensitivity=(f"- Deeper search changed exact selected site set in {changed}/{len(paired)} refined NVs; "
                            f"model order changed in {changed_order}/{len(paired)}.")
    lines=["# V8b analysis of the 52 G QNami spin-echo data","",
      f"Run: {path.name.removesuffix('_nv_summary.csv')}.  NVs: {len(s)}; points per NV: 94.",
      f"Search passes: {cfg.get('merged_search_passes', {'quick': len(s)})}.",
      f"Catalog orientation is fixed from ESR.  The exact independent spin-1/2 Hahn coherence product uses total time 2τ.",
      "The early 2.8/5.2 µs empirical transient is a fixed shared nuisance shape with one fitted amplitude per NV.",
      "It is not attributed to a lattice site.  Revival center is constrained to 34.5–36.8 µs.",
      "The optional amplitude taper and chirp are omitted because the pilot produced severe T2 degeneracy.","",
      "## Fit and model order","",
      f"- Numerical BIC rule (improvement ≥10 per added order): {orders}.  N=3 was not searched.",
      f"- Median reduced chi2: V8b {s.selected_red_chi2.median():.3f}; V6 {joined.v6_red.median():.3f}"+
        (f"; V7 {v7.best_red_chi2.median():.3f}." if v7 is not None else "."),
      f"- V8b beats V6 in reduced chi2 for {(joined.selected_red_chi2<joined.v6_red).sum()}/{len(joined)} NVs;"+
        f" within 10% of V6 for {(joined.selected_red_chi2<=1.1*joined.v6_red).sum()}/{len(joined)}.",
      f"- Poor absolute fit (reduced chi2 >3): {len(large)}/{len(s)}.  AICc/BIC among these are conditional diagnostics, not credible site evidence.","",
      "## Site inference","",
      f"- Numerically selected resolved sites: {len(selected)}. Adequate fit and ESR quality: {len(good)}."+
       f" Adequate with nonboundary visibility and ≥0.8 within-shortlist weight: {len(supported)}.",
      f"- Visibility at a boundary among selected: {int(eta.sum())}/{len(selected)}." if len(selected) else "- No resolved sites selected.",
      f"- Within-shortlist selected-site weight median: {selected.selected_akaike_weight_within_shortlist.median():.3f}." if len(selected) else "- Site weights unavailable for N=0.",
      f"- Selected-site kappa median: {selected.c13_1_kappa.median():.3f}." if len(selected) else "- No selected kappa.",
      "Site weights condition on the screened shortlist; they are not posterior probabilities of lattice identity.",
      "The catalog search multiplicity is not incorporated in ordinary AICc/BIC. Treat assignments as hypotheses.",
      search_sensitivity,family_sensitivity,
      f"- {len(sparse_alias_nv)} selected NVs have at least one site combination line above the 208 kHz Nyquist limit of the sparse 2.4-µs segments. The 65-point 23.664–47.664 µs dense segment has a 0.376-µs step (about 1.33 MHz Nyquist), so those lines rely heavily on its first-revival window.","",
      "## T2 and orientation","",
      f"- T2 point estimates median {s.T2_us.median():.1f} µs; range {s.T2_us.min():.1f}–{s.T2_us.max():.1f} µs.",
      f"- For the {(s.T2_inference_status=='bounded').sum()} adequate NVs with beta away from its bounds and a bounded T2 profile, median T2 is {s.loc[s.T2_inference_status=='bounded','T2_us'].median():.1f} µs (IQR {s.loc[s.T2_inference_status=='bounded','T2_us'].quantile(.25):.1f}–{s.loc[s.T2_inference_status=='bounded','T2_us'].quantile(.75):.1f} µs).",
      f"- T2 inference flags: {cutoff}.  Profile statuses: {s.T2_profile_status.value_counts().to_dict()}.",
      f"- At optimizer ceiling {len(upper)}; these are not reported as measured T2.",
      f"- Beta at a bound: {int(s.beta_boundary_hit.sum())}; early amplitude at a bound: {int(s.early_amp_boundary_hit.sum())}.",
      f"- ESR quality flags: {alt}; recorded measured and target frequencies, alternate RMS, and margin are in the CSV.",
      "No site list is ever searched from the alternative NV orientation.","",
      "## Representative rows",""]
    picks=list(dict.fromkeys([0,16,28,54,93,135,150,168,171]+
       s.nlargest(5,"selected_red_chi2").nv_index.astype(int).tolist()))
    cols=["nv_index","selected_order","selected_site_key","selected_red_chi2",
          "model_adequacy","site_inference_status","T2_us","T2_inference_status"]
    lines += [table(s[s.nv_index.isin(picks)].sort_values("nv_index"),cols,len(picks)),""]
    if cvfile.exists():
        cv=pd.read_csv(cvfile)
        pivot=cv.pivot_table(index="nv_index",columns="order",values="heldout_chi2_per_point")
        pivot.columns=[f"N{int(v)}" for v in pivot.columns]
        lines += ["## Later-time held-out prediction","",
          "Nested two-fold check: all points <15 µs stay in training; later points are interleaved and held out.",
          table(pivot.reset_index(),list(pivot.reset_index().columns),len(pivot)),"",
          "A lower held-out weighted chi2 per point predicts the omitted later echo data better.",""]
    stability=Path(base+"_block_residual_stability.csv")
    if stability.exists():
        boot=pd.read_csv(stability)
        bt=boot.groupby(["nv_index","selected_order"]).size().unstack(fill_value=0)
        bt.columns=[f"N{int(v)}" for v in bt.columns]
        bysite=boot.groupby(["nv_index","selected_sites"]).size().reset_index(name="count")
        top=bysite.sort_values(["nv_index","count"],ascending=[True,False]).drop_duplicates("nv_index")
        top["fraction"]=top["count"]/top.nv_index.map(boot.groupby("nv_index").size())
        lines += ["## Conditional block-residual stability","",
                  "Site models were refitted after circular five-point residual-block resampling within the saved site shortlist.",
                  table(bt.reset_index(),list(bt.reset_index().columns),len(bt)),"",
                  "Most frequent exact site set among replicates:",
                  table(top,["nv_index","selected_sites","count","fraction"],len(top)),"",
                  "Counts show numerical selections within a conditioned shortlist; stable order alone does not establish unique lattice identity.",""]
    lines += ["## Files and scope","",
      f"- Summary: {path.name}",
      f"- Candidate fits: {Path(base+'_candidate_fits.csv').name}",
      f"- Model orders: {Path(base+'_model_orders.csv').name}",
      f"- T2 profiles: {Path(base+'_t2_profiles.csv').name}",
      f"- Site families: {Path(base+'_site_families.csv').name}",
      f"- Coarse families: {family40.name}" if family40.exists() else "",
      f"- Conditional bootstrap: {stability.name}" if stability.exists() else "",
      f"- Diagnostic PDF: {Path(base+'_dashboard.pdf').name}",
      f"- Provenance: {Path(base+'_config.json').name}",
      "",
      "The pilot verified the Hahn propagator against an independent 2×2 matrix calculation.",
      "Residual structure, limited 83.32-µs evolution, unresolved 14N/control transients, and correlated point errors remain physical uncertainties.",
      "T2 is effective and conditional on the chosen bath envelope; precise intervals for poor fits or beta-bound cases should not be interpreted physically.",""]
    out=Path(base+"_report.md");out.write_text("\n".join(lines),encoding="utf-8")
    print(out)
if __name__=="__main__":main()
