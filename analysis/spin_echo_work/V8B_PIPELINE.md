# V8b: 52 G orientation-locked spin-echo pipeline

Run from the dioptric checkout on NVOffice:

```powershell
python analysis\spin_echo_work\sc_spin_echo_physics_fit_52G_v8b_multic13.py --max-spins 2 --quick --no-show --workers 12
```

The input discovery uses the latest matching V6 attempt checkpoint and the V6 ESR orientation assignment under `G:\nvdata\pc_NVOffice\branch_master`. It falls back to `\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master` when the drive is absent. The fixed lattice catalog is `essem_freq_kappa_catalog_22A_52G.json`. V6 supplies only background starting values; the isotope model is reoptimized against the 94 measured points.

Each completed NV is saved to a JSON checkpoint as soon as its search and T2 profile finish. Resume interrupted runs with the exact prior directory:

```powershell
python analysis\spin_echo_work\sc_spin_echo_physics_fit_52G_v8b_multic13.py --max-spins 2 --quick --no-show --workers 12 --resume-dir "\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09\v8b_20260923T060208287900Z_n212_quick_checkpoints"
```

## Physics and fit

`t` is total Hahn-echo evolution time (`2*tau`). A site contributes `1 - 2*eta*kappa*sin(pi*fI*t/2)^2*sin(pi*fm*t/2)^2`, and independent sites multiply. `eta` lies in `[0,1]`; a common timing shift lies in `[-0.25,0.25]` microseconds. The fixed ESR orientation restricts the catalog before any site search. A stretched exponential and quartic 35.66-microsecond revival carry the background. An empirically observed 2.8/5.2-microsecond shape has one fitted amplitude per NV and is never interpreted as carbon.

The search records N=0,1,2 candidates (N=3 only if explicitly requested) and accepts each additional order when BIC improves by at least 10. The quick pass fully reoptimizes 16 shortlisted singles and up to 20 pairs per NV. Omit `--quick` for the 36-single/90-pair deeper search; specify `--nv 4,27,...` to limit the run. `--no-profile` skips T2 profiling during exploratory passes.

Weighted least squares yields chi-squared, AICc, BIC, and a conditional T2 profile over 3-600 microseconds. The file `<run>_config.json` records inputs, hashes, bounds, catalog, model and time convention. `<run>_candidate_fits.csv`, `<run>_model_orders.csv`, `<run>_nv_summary.csv`, `<run>_t2_profiles.csv`, `<run>_dashboard.pdf`, and `<run>_global_summary.png/pdf` contain the results. The PDF has one diagnostic page per NV with the trace, first revival, conditional site weights, geometry, T2 profile, and fit table.

## Validation and interpretation

```powershell
python analysis\spin_echo_work\v8b_physics_checks.py
python analysis\spin_echo_work\v8b_heldout_validation.py --nv 0,16,28,54,93,135,150,168
python analysis\spin_echo_work\v8b_site_families.py --candidates "<run>_candidate_fits.csv"
python analysis\spin_echo_work\v8b_bootstrap_validation.py --candidate-csv "<run>_candidate_fits.csv" --nv 4,27,135,150 --reps 12 --workers 6
python analysis\spin_echo_work\v8b_summarize_results.py --summary "<run>_nv_summary.csv"
```

The held-out check refits the site search on the training points; the other positions at `t>=15` microseconds test prediction. Block-residual resampling tests candidate stability **within the saved shortlist**. Frequency families at 10 kHz bins expose near-degenerate sites. A numerical N>0 selection is only a conditional hypothesis: require an adequate absolute fit, reliable ESR orientation, nonboundary visibility, repeatable site preference, and useful held-out prediction before interpreting a lattice position. A profile extending to the 600-microsecond bound gives a lower limit, not a measured T2. The 94-point/83.32-microsecond window, learned empirical transient, catalog multiplicity, unmodeled pulse/bath effects, and possible correlated errors limit physical claims.

Catalog screening retains sites 1.4-22 Å from the NV, `kappa >= 1e-5`, an ESEEM combination line above 4 kHz, and the sampled Nyquist range. These are resolution/search choices, not evidence that excluded sites are physically absent. The results use a constrained 34.5-36.8-microsecond revival center and no revival amplitude taper or chirp. The common short-time shape was chosen after inspecting these data, so downstream evidence and intervals are conditional on that choice.

For a staged all-NV run, execute the quick 212-NV command above, then `v8b_select_refinements.py --summary "<quick>_nv_summary.csv"`. The selection records all numerical N>0 calls, up to ten N=0 cases with at least five units of candidate BIC gain, and representative controls. Copy its `NVS` list to `--nv` in a run without `--quick`. Finally execute `v8b_merge_refinements.py --quick-summary "<quick>_nv_summary.csv" --deep-summary "<deep>_nv_summary.csv"`; it verifies matching input/script hashes, replaces refined rows and regenerates a single 212-page dashboard. Run the validation/report commands above on the merged candidate and summary files.

The scan has a dense 65-point interval from 23.664 to 47.664 microseconds with 0.368/0.376-microsecond steps; outside it, some points are 1.6 or 2.4 microseconds apart. The catalog line filter uses the dense-window Nyquist limit. Lines above 208 kHz on the total-time axis are undersampled by the 2.4-microsecond portions and depend heavily on the dense revival interval. The report counts selected NVs in this category. Re-run `v8b_site_families.py --candidates "<run>_candidate_fits.csv" --bin-khz 40` to compare broader frequency families with the default 10-kHz bins; both are heuristic groupings within screened candidates.

## Completed 2026-09-23 run

The complete run prefix is `\\192.168.0.197\G\nvdata\pc_NVOffice\branch_master\sc_spin_echo_physics_fit_52G_nv_pillar_array\2026_09\v8b_final_20260923T062908790458Z_n212`. It combines 164 quick fits and 48 deeper fits. Read `<prefix>_report.md` first, inspect any NV in `<prefix>_dashboard.pdf`, and use `<prefix>_nv_summary.csv` plus `<prefix>_candidate_fits.csv` for analysis. The final PDF has 212 pages. The full held-out check is `v8b_heldout_validation_late.csv` in the same output directory. The conditional bootstrap and both frequency-family files share the final prefix.
