import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils import data_manager as dm
from utils import kplotlib as kpl
from typing import Optional
from collections.abc import Sequence
from typing import Iterable, Tuple, List, Dict, Optional

from analysis.sc_c13_hyperfine_sim_data_driven import (
    read_hyperfine_table_safe,
    B_vec_T,  # your lab field (Tesla)
)
from analysis.spin_echo_work.echo_fit_models import fine_decay, fine_decay_fixed_revival
from analysis.spin_echo_work.echo_plot_helpers import (
    extract_T2_freqs_and_errors,
    params_to_dict,
    plot_echo_with_sites,
    plot_branch_pairs,
    compare_two_fields,
    plot_branch_correlation_by_orientation,
    plot_c13_candidates_only,
)
from multiplicity_calculation import (
    find_c3v_orbits_from_nv2,
    build_site_multiplicity_with_theory,
    # mutliplicity_plots,
    multiplicity_plots,
    make_a_table,
)
from kappa_modulation_depth import (
    compute_dispersion_grid_for_site,
    plot_dispersion_2D_for_site,
    simulate_fpm_vs_B_for_site_all_fields,
    plot_site_sim_overlay,
    compute_full_angle_map_lab_spherical_for_site,
    plot_full_angle_map_theta_phi_lab,
    validate_site72_direct_vs_labmap,
)

# ---------------------------------------------------------------------
# CONFIG / PATHS
# ---------------------------------------------------------------------

HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
CATALOG_JSON = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_65G.json"
# CATALOG_JSON = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_49G.json"
# CATALOG_JSON = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_49G.json"

# Optional: which orientations to consider when comparing theory/exp
DEFAULT_ORIENTATIONS = [
    (1, 1, 1),
    (1, 1, -1),
    (1, -1, 1),
    (-1, 1, 1),
]
# ---------------------------------------------------------------------
# CATALOG + HYPERFINE LOADING
# ---------------------------------------------------------------------


def load_catalog(path_json: str = CATALOG_JSON):
    with open(path_json, "r") as f:
        return json.load(f)


def load_hyperfine_table(
    path_txt: str = HYPERFINE_PATH,
    distance_cutoff: float | None = None,
) -> pd.DataFrame:
    """
    Load the NV=2 hyperfine table and attach a `site_index` column that
    matches the indices used in the ESEEM catalog construction.

    Parameters
    ----------
    path_txt : str
        Path to the hyperfine tensor text file (NV-(111) frame).
    distance_cutoff : float or None
        If not None, keep only sites with `distance < distance_cutoff` (Å)
        *before* assigning `site_index`.

    Returns
    -------
    df : pandas.DataFrame
        Hyperfine table with a `site_index` column (0..N_filtered-1).
    """
    df = read_hyperfine_table_safe(path_txt).copy()

    if distance_cutoff is not None:
        df = df[df["distance"] < float(distance_cutoff)]

    df = df.reset_index(drop=True)
    df["site_index"] = df.index

    return df


def select_records(
    recs,
    fmin_kHz: float = 50.0,
    fmax_kHz: float = 6000.0,
    orientations=None,
):
    """
    Filter catalog records to a frequency window and (optionally) orientation set.
    """
    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
    else:
        ori_set = None

    out = []
    for r in recs:
        fm_k = r["f_minus_Hz"] / 1e3
        fp_k = r["f_plus_Hz"] / 1e3
        if not (np.isfinite(fm_k) and np.isfinite(fp_k)):
            continue
        if not (fmin_kHz <= fm_k <= fmax_kHz and fmin_kHz <= fp_k <= fmax_kHz):
            continue
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------
# EXPECTED SPECTRUM (THEORY)
# ---------------------------------------------------------------------
def lines_from_recs(
    recs,
    orientations=None,
    fmin_kHz: float = 50.0,
    fmax_kHz: float = 6000.0,
):
    """
    Return arrays: freqs_kHz (2 per site), weights (same length), and site_idx (pairs).
    Weights try to use physically motivated fields if present:
       - line_w_minus / line_w_plus
       - or amp_weight
       - or kappa
       - otherwise 1.
    """
    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
    else:
        ori_set = None

    freqs = []
    weights = []
    site_idx = []

    for i, r in enumerate(recs):
        if ori_set and tuple(r["orientation"]) not in ori_set:
            continue

        fm = r["f_minus_Hz"] / 1e3
        fp = r["f_plus_Hz"] / 1e3
        if not (np.isfinite(fm) and np.isfinite(fp)):
            continue
        if not (fmin_kHz <= fm <= fmax_kHz and fmin_kHz <= fp <= fmax_kHz):
            continue

        # Pick weight: prefer explicit line weights if present
        if "line_w_minus" in r and "line_w_plus" in r:
            w_minus = float(r["line_w_minus"])
            w_plus = float(r["line_w_plus"])
        else:
            base = float(r.get("amp_weight", r.get("kappa", 1.0)))
            w_minus = w_plus = base

        freqs.extend([fm, fp])
        weights.extend([w_minus, w_plus])
        site_idx.extend([r["site_index"], r["site_index"]])

    return np.array(freqs, float), np.array(weights, float), np.array(site_idx, int)


def expected_stick_spectrum_from_recs(
    recs,
    p_occ: float = 0.011,
    orientations=None,
    f_range_kHz=(50, 6000),
    use_weights: bool = True,
    merge_tol_kHz: float = 2.0,
    normalize: bool = False,
):
    """
    Build a discrete expected spectrum (stick plot) from catalog records.
    Returns: (f_stick_kHz, a_stick, fig, ax)
    """
    freqs, w, _ = lines_from_recs(
        recs,
        orientations=orientations,
        fmin_kHz=f_range_kHz[0],
        fmax_kHz=f_range_kHz[1],
    )
    if freqs.size == 0:
        raise ValueError("No catalog lines in the requested frequency range.")

    amps = p_occ * (w if use_weights else np.ones_like(w))

    order = np.argsort(freqs)
    f = freqs[order]
    a = amps[order]

    # merge nearby lines
    f_merged = []
    a_merged = []
    acc_f = f[0]
    acc_a = a[0]
    for f0, a0 in zip(f[1:], a[1:]):
        if abs(f0 - acc_f) <= merge_tol_kHz:
            new_a = acc_a + a0
            acc_f = (acc_f * acc_a + f0 * a0) / (new_a + 1e-30)
            acc_a = new_a
        else:
            f_merged.append(acc_f)
            a_merged.append(acc_a)
            acc_f, acc_a = f0, a0
    f_merged.append(acc_f)
    a_merged.append(acc_a)

    f_stick = np.asarray(f_merged, float)
    a_stick = np.asarray(a_merged, float)

    if normalize and a_stick.sum() > 0:
        a_stick = a_stick / a_stick.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xscale("log")
    ax.vlines(
        f_stick, 0.0, a_stick, linewidth=1.0, alpha=0.9, label="Expected (catalog)"
    )

    ax.set_xlim(*f_range_kHz)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Expected intensity (arb.)" + (" (normalized)" if normalize else ""))
    ax.set_title(f"Expected ESEEM stick spectrum (p_occ={p_occ:.3f})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(framealpha=0.85)
    return f_stick, a_stick, fig, ax


# ---------------------------------------------------------------------
# EXPERIMENTAL FREQUENCIES FROM FITS
# ---------------------------------------------------------------------


def extract_experimental_frequencies(
    file_stem: str,
    chi2_fail_thresh: float = 3.0,
    T2_thresh_us: float = 600.0,
):
    """
    Load fitted spin-echo data for a widefield dataset and return masked arrays
    for NV labels, T2, f0, f1, etc. using your existing extractor.
    """
    data = dm.get_raw_data(file_stem=file_stem)

    (
        nv,
        T2_us,
        f0_kHz,
        f1_kHz,
        A_pick_kHz,
        chis,
        fit_fail,
        sT2_us,
        sf0_kHz,
        sf1_kHz,
        sA_pick_kHz,
    ) = extract_T2_freqs_and_errors(
        data, pick_freq="max", chi2_fail_thresh=chi2_fail_thresh
    )

    nv = np.asarray(nv)

    # base validity
    valid = np.isfinite(T2_us) & (~fit_fail)
    # NOTE: ≥ threshold (long-lived spins), not ≤
    mask = valid & (T2_us <= T2_thresh_us)

    # masked arrays
    out = {
        "nv": nv[mask],
        "T2_us": np.asarray(T2_us)[mask],
        "sT2_us": np.asarray(sT2_us)[mask],
        "A_pick_kHz": np.asarray(A_pick_kHz)[mask],
        "sA_pick_kHz": np.asarray(sA_pick_kHz)[mask],
        "f0_kHz": np.asarray(f0_kHz)[mask],
        "f1_kHz": np.asarray(f1_kHz)[mask],
        "sf0_kHz": np.asarray(sf0_kHz)[mask],
        "sf1_kHz": np.asarray(sf1_kHz)[mask],
        "chis": np.asarray(chis)[mask],
    }
    return out


def plot_sorted_exp_branches(
    f0_kHz,
    f1_kHz,
    sf0_kHz=None,
    sf1_kHz=None,
    title_prefix="Experimental",
    f_range_kHz=(50, 6000),
):
    """
    Plot experimental f0 and f1 separately, sorted by frequency.
    """
    f0 = np.asarray(f0_kHz, float)
    f1 = np.asarray(f1_kHz, float)

    # branch 0
    mask0 = np.isfinite(f0) & (f0 >= f_range_kHz[0]) & (f0 <= f_range_kHz[1])
    order0 = np.argsort(f0[mask0])
    x0 = np.arange(1, order0.size + 1)
    y0 = f0[mask0][order0]

    # branch 1
    mask1 = np.isfinite(f1) & (f1 >= f_range_kHz[0]) & (f1 <= f_range_kHz[1])
    order1 = np.argsort(f1[mask1])
    x1 = np.arange(1, order1.size + 1)
    y1 = f1[mask1][order1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x0, y0, "o", ms=3, alpha=0.9, label="f0 (exp)")
    ax.plot(x1, y1, "o", ms=3, alpha=0.9, label="f1 (exp)")

    ax.set_yscale("log")
    ax.set_ylim(*f_range_kHz)
    ax.set_xlabel("NV index (sorted within branch)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title(f"{title_prefix}: f0 / f1 (sorted)")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# MATCHING EXPERIMENTAL (f0,f1) TO CATALOG SITES
# --------------------------------------------------------------------
def enrich_matches_with_hyperfine(matches_df, hf_df):
    """
    Merge in hyperfine-tensor info from the original NV-2 table.
    """
    out = matches_df.merge(
        hf_df,
        on="site_index",
        how="left",
        suffixes=("", "_hf"),
    )
    return out


def _build_catalog_arrays_all_orientations_kHz(
    records,
    orientations=None,
    kmin_kHz=1.0,
    kmax_kHz=6000.0,
):
    """
    Build arrays over all catalog sites (optionally filtered by orientation):

      f_plus_kHz, f_minus_kHz, distance_A, kappa,
      ori_vecs, site_idx, x_A, y_A, z_A
    """
    if orientations is not None:
        ori_set = {tuple(o) for o in orientations}
    else:
        ori_set = None

    fplus_list, fminus_list = [], []
    dist_list, kappa_list = [], []
    ori_list = []
    site_idx = []
    x_list, y_list, z_list = [], [], []

    for i, r in enumerate(records):
        ori_r = tuple(r.get("orientation", ()))
        if ori_set is not None and ori_r not in ori_set:
            continue

        fplus_Hz = r.get("f_plus_Hz", None)
        fminus_Hz = r.get("f_minus_Hz", None)
        if fplus_Hz is None or fminus_Hz is None:
            continue

        fplus_kHz = float(fplus_Hz) / 1e3
        fminus_kHz = float(fminus_Hz) / 1e3

        # both branches inside band
        if not (
            kmin_kHz <= fplus_kHz <= kmax_kHz and kmin_kHz <= fminus_kHz <= kmax_kHz
        ):
            continue

        fplus_list.append(fplus_kHz)
        fminus_list.append(fminus_kHz)
        dist_list.append(float(r.get("distance_A", np.nan)))
        kappa_list.append(float(r.get("kappa", np.nan)))
        ori_list.append(ori_r)
        site_idx.append(i)

        x_list.append(float(r.get("x_A", np.nan)))
        y_list.append(float(r.get("y_A", np.nan)))
        z_list.append(float(r.get("z_A", np.nan)))

    if not fplus_list:
        return (
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
            np.zeros((0, 3), int),
            np.asarray([], int),
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
        )

    return (
        np.asarray(fplus_list, float),
        np.asarray(fminus_list, float),
        np.asarray(dist_list, float),
        np.asarray(kappa_list, float),
        np.asarray(ori_list, int),
        np.asarray(site_idx, int),
        np.asarray(x_list, float),
        np.asarray(y_list, float),
        np.asarray(z_list, float),
    )


# ---------------------------------------------------------------------
# MAIN DRIVER EXAMPLE
# ---------------------------------------------------------------------
def pairwise_match_from_site_ids_kHz(
    nv_labels,
    f0_kHz,
    f1_kHz,
    site_ids,
    records,
    *,
    nv_orientations=None,  # per-NV orientation from fit file
    site_id_key: str = "site_index",
    fplus_key: str = "f_plus_Hz",
    fminus_key: str = "f_minus_Hz",
    verbose: bool = False,
):
    """
    Match NVs to 13C sites *using the site ID already encoded in the fit*.

    Logic:
      - For each NV, read its `site_id`.
      - Look up all catalog records with that `site_id`.
      - If an NV orientation is provided, pick the record whose catalog
        'orientation' matches that NV orientation.
      - Otherwise (or if no orientation match exists), fall back to the first
        record for that site_id.
      - Pull out catalog (f_plus_Hz, f_minus_Hz), assign them to (f0,f1)
        by ordering and compute residuals in kHz.

    If `nv_orientations` is provided (shape (N,3)), that orientation
    (e.g. (1,1,-1)) is stored in the output `orientation` column.

    Returns
    -------
    matches_df : pandas.DataFrame
        One row per NV with columns:

        nv_label, f0_fit_kHz, f1_fit_kHz,
        orientation, site_index, x_A, y_A, z_A, distance_A, kappa,
        f_plus_kHz, f_minus_kHz, assignment,
        err_pair_kHz, err_f0_kHz, err_f1_kHz, site_id
    """

    nv_labels = np.asarray(nv_labels)
    f0_kHz = np.asarray(f0_kHz, float)
    f1_kHz = np.asarray(f1_kHz, float)
    site_ids = np.asarray(site_ids)

    if f0_kHz.shape != f1_kHz.shape:
        raise ValueError("f0_kHz and f1_kHz must have same shape.")
    if f0_kHz.ndim != 1:
        raise ValueError("f0_kHz and f1_kHz must be 1D.")
    if site_ids.shape[0] != f0_kHz.shape[0]:
        raise ValueError("site_ids must have same length as f0_kHz/f1_kHz.")

    if nv_orientations is not None:
        nv_orientations = np.asarray(nv_orientations, int)
        if nv_orientations.shape[0] != f0_kHz.shape[0]:
            raise ValueError("nv_orientations must have same length as f0_kHz/f1_kHz.")

    # ------------------------------------------------------------------
    # Build a lookup from site_id -> list of catalog records
    # (because catalog has separate entries for different orientations)
    # ------------------------------------------------------------------
    site_lookup = {}
    for r in records:
        sid = r.get(site_id_key, None)
        if sid is None:
            continue
        sid_int = int(sid)
        site_lookup.setdefault(sid_int, []).append(r)

    rows = []
    N = f0_kHz.size

    for i in range(N):
        lbl = nv_labels[i] if i < nv_labels.size else i
        f0 = float(f0_kHz[i])
        f1 = float(f1_kHz[i])
        sid = site_ids[i]

        # orientation from fit (if provided)
        ori_fit = None
        if nv_orientations is not None:
            o = nv_orientations[i]
            try:
                ori_fit = tuple(int(v) for v in np.ravel(o))
            except Exception:
                ori_fit = tuple(np.ravel(o))

        # Basic sanity: invalid freqs → no match
        if not (np.isfinite(f0) and np.isfinite(f1)):
            rows.append(
                dict(
                    nv_label=int(lbl),
                    f0_fit_kHz=f0,
                    f1_fit_kHz=f1,
                    site_id=int(sid) if np.isfinite(sid) else -1,
                    site_index=-1,
                    orientation=ori_fit,
                    x_A=np.nan,
                    y_A=np.nan,
                    z_A=np.nan,
                    f_plus_kHz=np.nan,
                    f_minus_kHz=np.nan,
                    assignment="none",
                    err_pair_kHz=np.nan,
                    err_f0_kHz=np.nan,
                    err_f1_kHz=np.nan,
                    distance_A=np.nan,
                    kappa=np.nan,
                )
            )
            continue

        sid_int = int(sid) if np.isfinite(sid) else -1
        cand_list = site_lookup.get(sid_int, None)

        if not cand_list:
            # Site id not found in catalog → no match
            if verbose:
                print(f"NV {lbl}: site_id={sid_int} not found in catalog.")
            rows.append(
                dict(
                    nv_label=int(lbl),
                    f0_fit_kHz=f0,
                    f1_fit_kHz=f1,
                    site_id=sid_int,
                    site_index=-1,
                    orientation=ori_fit,
                    x_A=np.nan,
                    y_A=np.nan,
                    z_A=np.nan,
                    f_plus_kHz=np.nan,
                    f_minus_kHz=np.nan,
                    assignment="none",
                    err_pair_kHz=np.nan,
                    err_f0_kHz=np.nan,
                    err_f1_kHz=np.nan,
                    distance_A=np.nan,
                    kappa=np.nan,
                )
            )
            continue

        # ------------------------------------------------------------------
        # If we know the NV orientation, try to select the catalog record
        # whose "orientation" exactly matches ori_fit. If none match (or
        # ori_fit is None), fall back to the first record for that site_id.
        # ------------------------------------------------------------------
        chosen_rec = None
        chosen_rec_ori_tuple = None

        if ori_fit is not None:
            for r in cand_list:
                rec_ori = r.get("orientation", None)
                if rec_ori is None:
                    continue
                try:
                    rec_ori_tuple = tuple(int(v) for v in rec_ori)
                except Exception:
                    rec_ori_tuple = tuple(rec_ori)
                if rec_ori_tuple == ori_fit:
                    chosen_rec = r
                    chosen_rec_ori_tuple = rec_ori_tuple
                    break

        if chosen_rec is None:
            # No orientation match or no orientation info → use first record
            chosen_rec = cand_list[0]
            rec_ori = chosen_rec.get("orientation", None)
            if rec_ori is not None:
                try:
                    chosen_rec_ori_tuple = tuple(int(v) for v in rec_ori)
                except Exception:
                    chosen_rec_ori_tuple = tuple(rec_ori)

        rec = chosen_rec

        # --- Pull out catalog frequencies and metadata ---
        fplus_Hz = float(rec[fplus_key])
        fminus_Hz = float(rec[fminus_key])
        fplus_kHz = fplus_Hz * 1e-3
        fminus_kHz = fminus_Hz * 1e-3

        # Decide which catalog line is f0 vs f1
        if fplus_kHz >= fminus_kHz:
            assignment = "f0->f+, f1->f-"
            err_f0 = f0 - fplus_kHz
            err_f1 = f1 - fminus_kHz
        else:
            assignment = "f0->f-, f1->f+"
            err_f0 = f0 - fminus_kHz
            err_f1 = f1 - fplus_kHz

        err_pair = float(np.hypot(err_f0, err_f1))

        # Orientation for output: prefer fit orientation, else catalog
        out_ori = ori_fit if ori_fit is not None else chosen_rec_ori_tuple

        # Optional: warn if we *had* an orientation and we fell back
        if verbose and (ori_fit is not None) and (chosen_rec_ori_tuple is not None):
            if ori_fit != chosen_rec_ori_tuple:
                print(
                    f"[WARN] NV {lbl}: orientation mismatch or no exact match; "
                    f"fit={ori_fit}, chosen catalog ori={chosen_rec_ori_tuple}, "
                    f"site_id={sid_int}"
                )

        rows.append(
            dict(
                nv_label=int(lbl),
                f0_fit_kHz=f0,
                f1_fit_kHz=f1,
                site_id=sid_int,
                orientation=out_ori,
                site_index=int(rec.get("site_index", sid_int)),
                x_A=float(rec.get("x_A", np.nan)),
                y_A=float(rec.get("y_A", np.nan)),
                z_A=float(rec.get("z_A", np.nan)),
                distance_A=float(rec.get("distance_A", np.nan)),
                kappa=float(rec.get("kappa", np.nan)),
                f_plus_kHz=fplus_kHz,
                f_minus_kHz=fminus_kHz,
                assignment=assignment,
                err_pair_kHz=err_pair,
                err_f0_kHz=float(err_f0),
                err_f1_kHz=float(err_f1),
            )
        )

        if verbose:
            print(
                f"NV {lbl}, site_id={sid_int}, ori={out_ori}: "
                f"f0={f0:.2f}, f1={f1:.2f} kHz; "
                f"f+={fplus_kHz:.2f}, f-={fminus_kHz:.2f} kHz; "
                f"err_pair={err_pair:.2f} kHz, {assignment}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# COMBINED PLOT: EXPECTED SPECTRUM + EXP FREQUENCIES
# ---------------------------------------------------------------------


def plot_expected_with_exp_overlay(
    recs,
    exp_f0_kHz,
    exp_f1_kHz,
    matches_df=None,
    p_occ: float = 0.011,
    f_range_kHz=(50, 6000),
    orientations=None,
    merge_tol_kHz: float = 2.0,
):
    """
    - Plot expected stick spectrum from catalog (log-x).
    - Overlay experimental f0/f1 as vertical lines:
        * one color for matched
        * another for unmatched (if matches_df provided)
    """
    f_stick, a_stick, fig, ax = expected_stick_spectrum_from_recs(
        recs,
        p_occ=p_occ,
        orientations=orientations,
        f_range_kHz=f_range_kHz,
        merge_tol_kHz=merge_tol_kHz,
        normalize=False,
    )

    exp_f0 = np.asarray(exp_f0_kHz, float)
    exp_f1 = np.asarray(exp_f1_kHz, float)

    if matches_df is None:
        # simple overlay: all experimental lines same style
        for f in np.concatenate([exp_f0, exp_f1]):
            if f_range_kHz[0] <= f <= f_range_kHz[1]:
                ax.axvline(f, ymax=0.8, linestyle="--", alpha=0.5, color="C3")
        ax.plot([], [], "--", color="C3", alpha=0.8, label="Experimental f0,f1")
        ax.legend(framealpha=0.85)
        fig.tight_layout()
        return fig, ax

    # If we have matches_df, use it to separate matched vs unmatched NVs
    match_map = {int(row.nv): bool(row.has_match) for _, row in matches_df.iterrows()}

    matched_f = []
    unmatched_f = []
    for nv_label, f0, f1 in zip(matches_df["nv"], exp_f0, exp_f1):
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if match_map.get(int(nv_label), False):
            matched_f.extend([f0, f1])
        else:
            unmatched_f.extend([f0, f1])

    for f in matched_f:
        if f_range_kHz[0] <= f <= f_range_kHz[1]:
            ax.axvline(f, ymax=0.8, linestyle="--", alpha=0.8, color="C3")
    for f in unmatched_f:
        if f_range_kHz[0] <= f <= f_range_kHz[1]:
            ax.axvline(f, ymax=0.8, linestyle=":", alpha=0.6, color="C1")

    ax.plot([], [], "--", color="C3", alpha=0.9, label="Exp (matched NVs)")
    ax.plot([], [], ":", color="C1", alpha=0.9, label="Exp (unmatched NVs)")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    return fig, ax


# -------------------------------------------------------------------
# 1) σ-distance between an experimental pair (f0,f1) and catalog (fm,fp)
# -------------------------------------------------------------------
def pair_distance_sigma_vec(
    f0_kHz: float,
    f1_kHz: float,
    sf0_kHz: float,
    sf1_kHz: float,
    fm_kHz: np.ndarray,
    fp_kHz: np.ndarray,
) -> np.ndarray:
    """
    Return the per-site σ-distance between experimental (f0,f1) and
    catalog (fm,fp), taking into account both labelings:
      (f0 ↔ fm, f1 ↔ fp) and (f0 ↔ fp, f1 ↔ fm)

    D_j = min( sqrt(((f0-fm_j)/sf0)^2 + ((f1-fp_j)/sf1)^2),
               sqrt(((f0-fp_j)/sf0)^2 + ((f1-fm_j)/sf1)^2) )
    """
    fm_kHz = np.asarray(fm_kHz, float)
    fp_kHz = np.asarray(fp_kHz, float)

    d1 = np.sqrt(
        ((f0_kHz - fm_kHz) / sf0_kHz) ** 2 + ((f1_kHz - fp_kHz) / sf1_kHz) ** 2
    )
    d2 = np.sqrt(
        ((f0_kHz - fp_kHz) / sf0_kHz) ** 2 + ((f1_kHz - fm_kHz) / sf1_kHz) ** 2
    )
    return np.minimum(d1, d2)


def pair_distance_plain_vec(
    f0_kHz: float,
    f1_kHz: float,
    fm_kHz: np.ndarray,
    fp_kHz: np.ndarray,
) -> np.ndarray:
    """
    Plain Euclidean distance in kHz between (f0,f1) and (fm,fp),
    again minimizing over the two possible assignments.
    """
    fm_kHz = np.asarray(fm_kHz, float)
    fp_kHz = np.asarray(fp_kHz, float)

    d1 = np.sqrt((f0_kHz - fm_kHz) ** 2 + (f1_kHz - fp_kHz) ** 2)
    d2 = np.sqrt((f0_kHz - fp_kHz) ** 2 + (f1_kHz - fm_kHz) ** 2)
    return np.minimum(d1, d2)


# -------------------------------------------------------------------
# 2) Main analysis: NV ↔ 13C site matching with confidence metrics
# -------------------------------------------------------------------
_fn_map = {
    "fine_decay": fine_decay,
    "fine_decay_fixed_revival": fine_decay_fixed_revival,
}


def _get_coord_cols(hf_df: pd.DataFrame):
    """
    Detect which columns hold NV-frame coordinates in Å.
    Adjust if your nv-2 table uses different names.
    """
    if {"x_A", "y_A", "z_A"}.issubset(hf_df.columns):
        return "x_A", "y_A", "z_A"
    if {"x", "y", "z"}.issubset(hf_df.columns):
        return "x", "y", "z"
    raise KeyError(
        "Could not find coordinate columns in hyperfine table; "
        "modify _get_coord_cols() to match your column names."
    )


def _find_catalog_rec_for_match(row, catalog_recs):
    """
    Find the catalog record corresponding to this match row.

    We match on:
      - site_index
      - orientation  (tuple of 3 ints)
    """
    if catalog_recs is None:
        return None

    # site_index
    site_index = int(row.get("site_index", -1))

    # Orientation: normalize to tuple[int,int,int]
    ori = row.get("orientation", None)
    if isinstance(ori, str):
        try:
            s = ori.strip("()[]")
            parts = [int(x) for x in s.replace(",", " ").split()]
            ori = tuple(parts) if len(parts) == 3 else None
        except Exception:
            ori = None
    elif isinstance(ori, (list, tuple)) and len(ori) == 3:
        ori = tuple(int(v) for v in ori)
    else:
        ori = None

    for rec in catalog_recs:
        if int(rec.get("site_index", -1)) != site_index:
            continue
        rec_ori = rec.get("orientation", None)
        if isinstance(rec_ori, (list, tuple)) and len(rec_ori) == 3:
            rec_ori = tuple(int(v) for v in rec_ori)
        else:
            rec_ori = None

        if ori is None or rec_ori is None:
            # if something is missing, at least match on site_index
            if int(rec.get("site_index", -1)) == site_index:
                return rec
        else:
            if rec_ori == ori:
                return rec

    return None


def _build_site_info_from_match_and_catalog(row, hf_row, catalog_recs):
    """
    row          : one row from matches_enriched (Series for single NV)
    hf_row       : matching hyperfine row (geometry, coords, etc.)
    catalog_recs : list of dicts from essem_freq_catalog_XX.json
    """
    cat_rec = _find_catalog_rec_for_match(row, catalog_recs)

    # ---- Geometry: r ----
    r_val = hf_row.get("r", np.nan)
    if not np.isfinite(r_val):
        r_val = row.get("distance_A", np.nan)

    # ---- Orientation ----
    if cat_rec is not None:
        ori = cat_rec.get("orientation", None)
    else:
        ori = row.get("orientation", None)

    if isinstance(ori, str):
        try:
            s = ori.strip("()[]")
            parts = [int(x) for x in s.replace(",", " ").split()]
            ori = tuple(parts) if len(parts) == 3 else None
        except Exception:
            ori = None
    elif isinstance(ori, (list, tuple)) and len(ori) == 3:
        ori = tuple(int(v) for v in ori)
    else:
        ori = None

    def _cat_float(key, default=np.nan, scale=1.0):
        if cat_rec is None or key not in cat_rec:
            return float(default)
        v = cat_rec[key]
        try:
            return float(v) * scale
        except Exception:
            return float(default)

    # A_par / A_perp from catalog if available, else matches_enriched
    A_par_kHz = _cat_float("A_par_Hz", default=np.nan, scale=1e-3)
    A_perp_kHz = _cat_float("A_perp_Hz", default=np.nan, scale=1e-3)

    if np.isnan(A_par_kHz) and "A_par_Hz" in row.index:
        try:
            A_par_kHz = float(row["A_par_Hz"]) * 1e-3
        except Exception:
            pass
    if np.isnan(A_perp_kHz) and "A_perp_Hz" in row.index:
        try:
            A_perp_kHz = float(row["A_perp_Hz"]) * 1e-3
        except Exception:
            pass

    # theta from catalog or row
    theta_deg = _cat_float("theta_deg", default=np.nan, scale=1.0)
    if np.isnan(theta_deg) and "theta_deg" in row.index:
        try:
            theta_deg = float(row["theta_deg"])
        except Exception:
            pass

    # kappa, fI, f-/f+ from catalog
    kappa = _cat_float("kappa", default=np.nan)
    fI_kHz = _cat_float("fI_Hz", default=np.nan, scale=1e-3)
    fm_kHz = _cat_float("f_minus_Hz", default=np.nan, scale=1e-3)
    fp_kHz = _cat_float("f_plus_Hz", default=np.nan, scale=1e-3)

    site_index = int(row.get("site_index", hf_row.get("site_index", -1)))

    return [
        {
            "site_id": site_index,
            "r": float(r_val),
            "Apar_kHz": float(A_par_kHz),
            "Aperp_kHz": float(A_perp_kHz),
            "theta_deg": float(theta_deg),
            "kappa": float(kappa),
            "fI_kHz": float(fI_kHz),
            "fm_kHz": float(fm_kHz),
            "fp_kHz": float(fp_kHz),
            "orientation": ori,
        }
    ]


def make_echo_plus_matched_site_plot(
    counts_file_stem: str,
    fit_file_stem: str,
    matches_enriched: pd.DataFrame,
    hf_df: Optional[pd.DataFrame],
    nv_label: int,
    use_half_time_as_tau: bool = True,
    units_label: str = "(Norm.)",
):
    """
    Make a single figure:
      left: experimental spin-echo trace + fit + envelope
      right: matched 13C site in 3D (NV frame)
    for a chosen NV label.

    If `hf_df` is None, only the matched site (from `matches_enriched`)
    will be shown (no background candidate cloud).
    """

    # ---------- 1) Match-row for this NV ----------
    row = matches_enriched.loc[matches_enriched["nv_label"] == nv_label]
    if row.empty:
        raise ValueError(f"No entry for NV {nv_label} in matches_enriched.")
    row = row.iloc[0]

    # ---------- 1b) Optional hyperfine table info (background cloud) ----------
    hf_row = None
    all_pos = None
    xcol = ycol = zcol = None

    stats = {
        "N_candidates": 0,
        "abundance_fraction": None,
    }

    if hf_df is not None and not hf_df.empty:
        # Figure out which columns are x/y/z in hf_df
        xcol, ycol, zcol = _get_coord_cols(hf_df)

        # Background: all candidate site positions
        if {xcol, ycol, zcol}.issubset(hf_df.columns):
            all_pos = hf_df[[xcol, ycol, zcol]].to_numpy(float)

        stats["N_candidates"] = len(hf_df)

        # Try to find a matching hyperfine row for this site_index
        site_index_val = row.get("site_index", np.nan)
        if np.isfinite(site_index_val):
            site_index = int(site_index_val)
            hf_row_candidates = hf_df.loc[hf_df["site_index"] == site_index]
            if hf_row_candidates.empty:
                print(
                    f"[WARN] No hyperfine row found for site_index={site_index}; "
                    "will still plot matched site if coordinates are in matches_enriched."
                )
            else:
                hf_row = hf_row_candidates.iloc[0]
        else:
            print(
                f"[WARN] NV {nv_label} has no valid site_index; "
                "will still plot matched site if coordinates are in matches_enriched."
            )
    else:
        # No hyperfine table: no background positions
        all_pos = None

    # ---------- 1c) Foreground: matched site position ----------
    matched_pos = None
    site_info = []

    # Prefer coordinates stored directly in the matched row
    x = row.get("x_A", np.nan)
    y = row.get("y_A", np.nan)
    z = row.get("z_A", np.nan)

    if np.all(np.isfinite([x, y, z])):
        matched_pos = np.array([[x, y, z]], dtype=float)

    # If we have a hyperfine row and site_info helper, build site_info
    if hf_row is not None:
        catalog_recs = load_catalog(CATALOG_JSON)
        site_info = _build_site_info_from_match_and_catalog(row, hf_row, catalog_recs)

        # Optional: if matched_pos is still None, fall back to hf_row coords
        if matched_pos is None and (xcol is not None):
            matched_pos = np.array(
                [[hf_row[xcol], hf_row[ycol], hf_row[zcol]]],
                dtype=float,
            )

    if matched_pos is None:
        print(
            f"[WARN] NV {nv_label}: no valid coordinates for matched site; "
            "will plot echo only."
        )

    # ---------- 2) Load counts data (echo) ----------
    data_counts = dm.get_raw_data(file_stem=counts_file_stem)

    norm_counts = data_counts["norm_counts"]
    norm_counts_ste = data_counts["norm_counts_ste"]
    total_times_us = np.asarray(data_counts["total_evolution_times"], float)

    echo = np.asarray(norm_counts[int(nv_label)], float)
    echo_ste = np.asarray(norm_counts_ste[int(nv_label)], float)

    if use_half_time_as_tau:
        taus_us = total_times_us / 2.0
    else:
        taus_us = total_times_us

    # ---------- 3) Load fit data (parameters) ----------
    data_fit = dm.get_raw_data(file_stem=fit_file_stem)

    fit_nv_labels = np.array(list(map(int, data_fit["nv_labels"])))
    idx = np.where(fit_nv_labels == int(nv_label))[0]
    if idx.size == 0:
        raise ValueError(
            f"NV {nv_label} not found in fit_nv_labels for {fit_file_stem}."
        )
    idx = int(idx[0])

    popts = data_fit["popts"]
    fit_fn_names = data_fit["fit_fn_names"]

    fit_fn_name = fit_fn_names[idx]
    fit_fn = _fn_map.get(fit_fn_name, fine_decay)
    p = np.asarray(popts[idx], float)

    fine_params = params_to_dict(fit_fn, p, default_rev=39.2)
    if "T2_fit_us" not in fine_params or fine_params["T2_fit_us"] is None:
        fine_params["T2_fit_us"] = 1000.0 * fine_params.get("T2_ms", 0.0)

    # ---------- 4) Build aux dict for plot_echo_with_sites ----------
    aux = {
        "positions": matched_pos,
        "site_info": site_info,
        "all_candidate_positions": all_pos,
        "stats": stats,
        "distance_cutoff": float(row.get("distance_A", np.nan)),
        "Ak_min_kHz": None,
        "Ak_max_kHz": None,
        "picked_ids_per_realization": None,
    }

    sim_info = {
        "T2_fit_us": fine_params.get("T2_fit_us", None),
        "distance_cutoff": aux["distance_cutoff"],
    }

    # ---------- 5) Call your existing helper ----------
    fig = plot_echo_with_sites(
        taus_us=taus_us,
        echo=echo,
        aux=aux,
        title="Spin Echo (experimental + matched site)",
        rmax=None,
        fine_params=fine_params,
        units_label=units_label,
        nv_label=nv_label,
        sim_info=sim_info,
        show_env=False,
        show_env_times_comb=False,
        echo_ste=echo_ste,
        fit_fn=fit_fn,
        fit_params=p,
        tau_is_half_time=use_half_time_as_tau,
        default_rev_for_plot=39.2,
    )
    plot_c13_candidates_only(aux)
    return fig


def make_echo_plus_matched_site_plots_batch(
    counts_file_stem: str,
    fit_file_stem: str,
    matches_enriched: pd.DataFrame,
    hf_df: Optional[pd.DataFrame],
    nv_labels: Sequence[int],
    use_half_time_as_tau: bool = True,
    units_label: str = "(Norm.)",
):
    """
    Generate one echo+matched-site figure per NV in `nv_labels`.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of figures, one per NV.
    """
    figs = []
    for nv_label in nv_labels:
        fig = make_echo_plus_matched_site_plot(
            counts_file_stem=counts_file_stem,
            fit_file_stem=fit_file_stem,
            matches_enriched=matches_enriched,
            hf_df=hf_df,
            nv_label=int(nv_label),
            use_half_time_as_tau=use_half_time_as_tau,
            units_label=units_label,
        )
        figs.append(fig)
    return figs


_fn_map = {
    "fine_decay": fine_decay,
    "fine_decay_fixed_revival": fine_decay_fixed_revival,
}


def freqs_from_popts_exact(
    file_stem: str,
    default_rev: float = 39.2,
):
    data_fit = dm.get_raw_data(file_stem=file_stem)

    nv_labels = np.array(list(map(int, data_fit["nv_labels"])))
    popts_list = data_fit["popts"]  # list/array of per-NV popt
    fit_fn_names = data_fit["fit_fn_names"]
    chis = np.asarray(data_fit.get("chis", np.nan), float)

    N = len(nv_labels)

    # ---- orientations + site_ids straight from fit file ----
    nv_orientations = None
    if "orientations" in data_fit:
        nv_orientations = np.asarray(data_fit["orientations"], int)
        if nv_orientations.shape[0] != N:
            raise ValueError(
                f"orientations length {nv_orientations.shape[0]} "
                f"does not match nv_labels length {N}"
            )

    site_ids = None
    if "site_id" in data_fit:
        site_ids = np.asarray(data_fit["site_id"], int)
        if site_ids.shape[0] != N:
            raise ValueError(
                f"site_id length {site_ids.shape[0]} "
                f"does not match nv_labels length {N}"
            )

    # --- allocate outputs ---
    T2_us = np.full(N, np.nan, float)
    f0_kHz = np.full(N, np.nan, float)
    f1_kHz = np.full(N, np.nan, float)
    A_pick_kHz = np.full(N, np.nan, float)
    sT2_us = np.full(N, np.nan, float)
    sf0_kHz = np.full(N, np.nan, float)
    sf1_kHz = np.full(N, np.nan, float)
    sA_pick_kHz = np.full(N, np.nan, float)
    fit_fail = np.zeros(N, bool)

    for i in range(N):
        name = fit_fn_names[i]
        fn = _fn_map.get(name, fine_decay)

        p_raw = popts_list[i]

        # --------- GUARD: bad / missing popts ----------
        if p_raw is None:
            fit_fail[i] = True
            continue

        p_arr = np.asarray(p_raw, float)
        if (p_arr.ndim == 0) or (p_arr.size < 2) or (not np.all(np.isfinite(p_arr))):
            fit_fail[i] = True
            continue

        p = p_arr
        par = params_to_dict(fn, p, default_rev=default_rev)

        # ---- frequencies directly from model ----
        f0 = par.get("osc_f0_kHz", par.get("f0_kHz", None))
        f1 = par.get("osc_f1_kHz", par.get("f1_kHz", None))

        # fallback: cycles/µs → MHz → kHz
        if f0 is None:
            osc_f0 = par.get("osc_f0", None)
            if osc_f0 is not None:
                f0 = 1e3 * float(osc_f0)
        if f1 is None:
            osc_f1 = par.get("osc_f1", None)
            if osc_f1 is not None:
                f1 = 1e3 * float(osc_f1)

        if f0 is not None:
            f0_kHz[i] = float(f0)
        if f1 is not None:
            f1_kHz[i] = float(f1)

        # ---- T2 ----
        T2 = par.get("T2_fit_us", None)
        if T2 is None and "T2_ms" in par:
            T2 = 1000.0 * par["T2_ms"]
        if T2 is not None:
            T2_us[i] = float(T2)

        # ---- amplitude for kappa consistency ----
        A_pick = (
            par.get("A_pick_kHz") or par.get("Ak_pick_kHz") or par.get("Ak_eff_kHz")
        )
        if A_pick is not None:
            A_pick_kHz[i] = float(A_pick)

        # errors: still NaN for now
        sT2_us[i] = np.nan
        sf0_kHz[i] = np.nan
        sf1_kHz[i] = np.nan
        sA_pick_kHz[i] = np.nan

        # ---- optional chi²-based fail flag ----
        if chis is None:
            chi = np.nan
        elif chis.ndim == 0:
            chi = float(chis)
        elif i < chis.size:
            chi = float(chis[i])
        else:
            chi = np.nan

        if np.isfinite(chi) and chi > 5.0:
            fit_fail[i] = True

    out = dict(
        nv=nv_labels,
        T2_us=T2_us,
        f0_kHz=f0_kHz,
        f1_kHz=f1_kHz,
        A_pick_kHz=A_pick_kHz,
        chis=chis,
        fit_fail=fit_fail,
        sT2_us=sT2_us,
        sf0_kHz=sf0_kHz,
        sf1_kHz=sf1_kHz,
        sA_pick_kHz=sA_pick_kHz,
    )

    # attach orientations / site_ids if present
    if nv_orientations is not None:
        out["orientations"] = nv_orientations
    if site_ids is not None:
        out["site_ids"] = site_ids

    return out


def attach_equiv_multiplicity(site_stats, c13_abundance=0.011, min_frac=0.01):
    """
    Given `site_stats` (from analyze_matched_c13_sites),
    add an estimate of the effective number of symmetry-equivalent positions
    per NV (N_equiv) for each 13C site, assuming random 13C occupancy and
    approximate detection efficiency.

      p_occ ≈ frac_NV ≈ 1 - (1 - f)^{N_equiv}
      ⇒ N_equiv ≈ ln(1 - frac_NV) / ln(1 - f)

    Parameters
    ----------
    site_stats : DataFrame
        Must contain columns:
          - 'frac_NV'   (fraction of NVs matched to this site)
          - 'orientation', 'site_index', 'distance_A', x_A, y_A, z_A, ...
    c13_abundance : float
        13C fraction f (≈ 0.011 for natural abundance).
    min_frac : float
        Minimum frac_NV to attempt a multiplicity estimate.
        (Below this, noise dominates and the estimate is meaningless.)

    Returns
    -------
    site_stats_with_N : DataFrame
        Copy of `site_stats` with a new column 'N_equiv_est'.
    """
    site_stats = site_stats.copy()
    f = float(c13_abundance)

    # default: NaN
    site_stats["N_equiv_est"] = np.nan

    # where frac_NV is big enough to be meaningful
    good = (site_stats["frac_NV"] > min_frac) & (site_stats["frac_NV"] < 0.99)

    frac = site_stats.loc[good, "frac_NV"].to_numpy(float)
    N_est = np.log(1.0 - frac) / np.log(1.0 - f)

    site_stats.loc[good, "N_equiv_est"] = N_est

    # print a quick summary of the most "symmetric" shells
    print("\nTop candidate symmetry shells (by N_equiv_est):")
    cols_show = [
        "orientation",
        "site_index",
        "distance_A",
        "x_A",
        "y_A",
        "z_A",
        "n_matches",
        "frac_NV",
        "N_equiv_est",
    ]
    print(
        site_stats.sort_values("N_equiv_est", ascending=False)
        .loc[site_stats["N_equiv_est"].notna(), cols_show]
        .head(15)
        .to_string(index=False)
    )

    return site_stats


def plot_multiplicity_hist(site_stats, title_prefix="Matched 13C sites"):
    """
    Histogram of how many sites have N matches.
    Uses `n_matches` from site_stats.
    """
    n = site_stats["n_matches"].to_numpy(int)

    values, counts = np.unique(n, return_counts=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(values, counts, width=0.7, align="center")

    ax.set_xlabel("n_matches (how many NVs matched this site)")
    ax.set_ylabel("Number of 13C sites")
    ax.set_title(f"{title_prefix}: site multiplicity histogram")

    for v, c in zip(values, counts):
        ax.text(v, c + 0.5, str(c), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_distance_vs_Nequiv(site_stats, title_prefix="Matched 13C sites"):
    """
    Scatter of distance_A vs N_equiv_est, colored by n_matches.
    Only shows rows with finite N_equiv_est.
    """
    df = site_stats[np.isfinite(site_stats["N_equiv_est"])].copy()
    if df.empty:
        print("[WARN] No finite N_equiv_est to plot.")
        return None, None

    fig, ax = plt.subplots(figsize=(6, 4.5))

    sc = ax.scatter(
        df["distance_A"],
        df["N_equiv_est"],
        c=df["n_matches"],
        s=30,
        alpha=0.9,
    )

    ax.set_xlabel("distance (Å)")
    ax.set_ylabel("N_equiv_est (effective symmetry multiplicity)")
    ax.set_title(f"{title_prefix}: distance vs. multiplicity")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("n_matches (NVs per site)")

    return fig, ax


def plot_kappa_vs_distance(site_stats, title_prefix="Matched 13C sites"):
    """
    κ vs distance, colored by N_equiv_est, with point size ~ n_matches.
    """
    df = site_stats.copy()
    # some rows may not have kappa_mean (NaN) – filter them
    df = df[np.isfinite(df["kappa_mean"])]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    sizes = 10 + 4 * df["n_matches"]  # bigger = more NVs
    colors = df["N_equiv_est"].where(np.isfinite(df["N_equiv_est"]), np.nan)

    sc = ax.scatter(
        df["distance_A"],
        df["kappa_mean"],
        s=sizes,
        c=colors,
        alpha=0.9,
    )

    ax.set_xlabel("distance (Å)")
    ax.set_ylabel("κ (mean, from catalog)")
    ax.set_title(f"{title_prefix}: κ vs distance")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("N_equiv_est")

    return fig, ax


def plot_sites_3d_multiplicity(site_stats, title_prefix="Matched 13C sites"):
    """
    3D scatter of unique 13C sites (x_A, y_A, z_A),
    color = n_matches, size ~ N_equiv_est.
    """
    df = site_stats.copy()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    sizes = 10 + 4 * np.nan_to_num(df["N_equiv_est"], nan=1.0)

    sc = ax.scatter(
        df["x_A"],
        df["y_A"],
        df["z_A"],
        c=df["n_matches"],
        s=sizes,
        alpha=0.9,
    )

    ax.scatter(
        0.0,
        0.0,
        0.0,
        marker="*",
        s=60,
        edgecolor="k",
        linewidth=0.6,
        label="NV center",
    )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(f"{title_prefix}: 13C sites\n" "color = n_matches, size ∝ N_equiv_est")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Number of NVs matched (n_matches)")

    ax.legend(loc="upper left")
    return fig, ax


def plot_freq_vs_kappa(matches, title_prefix="Matched 13C sites"):
    """
    Plot best-matching ESEEM frequency vs kappa, colored by orientation.

    Expects columns:
      f_match_kHz, err_match_kHz, kappa, orientation
    """
    df = matches.copy()
    df = df[np.isfinite(df["f_match_kHz"]) & np.isfinite(df["kappa"])]

    if df.empty:
        print("[WARN] No finite f_match_kHz or kappa to plot.")
        return None, None

    df["ori_label"] = df["orientation"].apply(ori_to_str)

    fig, ax = plt.subplots(figsize=(6, 5))

    for ori_val, sub in df.groupby("orientation"):
        ori_lab = ori_to_str(ori_val)
        sc = ax.scatter(
            sub["kappa"],
            sub["f_match_kHz"],
            s=20,
            alpha=0.8,
            label=ori_lab,
        )

    ax.set_xlabel("κ (ESEEM weight)")
    ax.set_ylabel("Matched frequency (kHz)")
    ax.set_title(f"{title_prefix}: best ESEEM frequency vs κ", fontsize=15)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Orientation", fontsize=8)

    return fig, ax


def ori_to_str(ori):
    """
    Convert orientation (e.g. (-1, 1, 1)) to a compact string.
    """
    if ori is None or (isinstance(ori, float) and not np.isfinite(ori)):
        return "ori=None"
    try:
        o = tuple(int(v) for v in ori)
    except Exception:
        o = tuple(ori)
    return f"({o[0]},{o[1]},{o[2]})"


def analyze_matched_c13_sites(matches_df, *, title_prefix="Matched 13C sites"):
    """
    Given a matches_df returned by match_exp_pairs_to_catalog(...),
    make a set of summary plots:

      1) 3D scatter of all unique matched 13C lattice sites (x_A, y_A, z_A),
         colored by NV orientation.

      2) Distance vs. matched frequency (one frequency per NV, pick the
         branch with smaller error).

      3) κ (kappa) vs. distance for unique sites.

      4) Histogram of distance distribution, split by NV orientation.

      5) (Optional) sorted f0/f1 theory branches for quick sanity check.

    Expected columns in matches_df (from your matcher):
        nv
        f0_exp_kHz, f1_exp_kHz
        f0_theory_kHz, f1_theory_kHz
        err0_kHz, err1_kHz
        orientation
        site_index
        distance_A
        kappa
        x_A, y_A, z_A  (added when building the catalog)
    """

    # ------------------------------------------------------------------
    # 0) Basic copy
    # ------------------------------------------------------------------
    matches = matches_df.copy()

    # ------------------------------------------------------------------
    # 1) Collapse to unique sites (orientation + site_index)
    #    → 3D scatter of all unique matched lattice sites
    # ------------------------------------------------------------------
    site_cols = [
        "orientation",
        "site_index",
        "x_A",
        "y_A",
        "z_A",
        "distance_A",
        "kappa",
    ]

    # 0) Use *all* matches → one point per NV–site match (no drop_duplicates)
    sites = matches[site_cols].copy()

    # Counts for caption
    n_matches = len(sites)
    n_unique_sites = sites[["orientation", "site_index"]].drop_duplicates().shape[0]

    # # Columns that define a unique 13C lattice site
    # site_key = ["orientation", "site_index", "x_A", "y_A", "z_A", "distance_A"]

    # # Total matches and unique sites
    # n_matches = len(matches)
    # n_unique_sites = matches.drop_duplicates(subset=site_key).shape[0]

    # print(f"Total NV–site matches: {n_matches}")
    # print(f"Unique 13C sites:      {n_unique_sites}")

    # # Optional: string label for orientation (for plotting, legends, etc.)
    # matches["ori_label"] = matches["orientation"].astype(str)

    # Columns that define a unique 13C lattice site
    site_key = ["orientation", "site_index", "x_A", "y_A", "z_A", "distance_A"]

    # Group by unique site and count how many NVs hit each site
    site_stats = matches.groupby(site_key, as_index=False).agg(
        n_matches=("nv_label", "count"),
        kappa_mean=("kappa", "mean"),
    )

    n_total_matches = len(matches)
    n_unique_sites = len(site_stats)

    print(f"Total NV–site matches: {n_total_matches}")
    print(f"Unique 13C sites:      {n_unique_sites}")

    # Add orientation label for plotting
    site_stats["ori_label"] = site_stats["orientation"].apply(ori_to_str)

    # === 3D scatter of unique sites, SPLIT by orientation ===
    for ori_val, sub in site_stats.groupby("orientation"):
        ori_lab = ori_to_str(ori_val)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        p = ax.scatter(
            sub["x_A"],
            sub["y_A"],
            sub["z_A"],
            c=sub["n_matches"],  # color = count of matches at that site
            s=15,
            alpha=0.9,
        )
        ax.scatter(
            0.0,
            0.0,
            0.0,
            marker="*",
            s=60,
            edgecolor="k",
            linewidth=0.6,
            label="NV center",
        )

        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.set_title(
            f"{title_prefix}: 13C sites by match count\n"
            f"orientation = {ori_lab}, "
            f"{len(sub)} sites (out of {n_unique_sites} unique sites)",
            fontsize=15,
        )

        cbar = fig.colorbar(p, ax=ax, shrink=0.7)
        cbar.set_label("Number of NVs matched to this site", fontsize=15)

        ax.legend(loc="upper left")

    # 1) Optional: orientation as label if you ever want to use it
    sites["ori_label"] = sites["orientation"].apply(ori_to_str)

    for ori_val, sub in sites.groupby("orientation"):
        ori_lab = ori_to_str(ori_val)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        p = ax.scatter(
            sub["x_A"],
            sub["y_A"],
            sub["z_A"],
            c=sub["kappa"],  # color by kappa
            s=15,
            alpha=0.9,
        )
        ax.scatter(
            0.0,
            0.0,
            0.0,
            marker="*",
            s=60,
            edgecolor="k",
            linewidth=0.6,
            label="NV center",
        )

        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_zlabel("z (Å)")
        ax.set_title(
            f"{title_prefix}: matched 13C sites\n"
            f"orientation = {ori_lab}\n"
            f"({len(sub)} NV–site matches, {n_unique_sites} unique sites total)",
            fontsize=15,
        )

        cbar = fig.colorbar(p, ax=ax, shrink=0.7)
        cbar.set_label("κ (ESEEM weight)", fontsize=15)

        ax.legend(loc="upper left")

    # ------------------------------------------------------------------
    # 2) Add "fraction of NVs" for each site (for later back-of-envelope N_eq)
    # ------------------------------------------------------------------
    n_nv = matches["nv_label"].nunique()
    site_stats["frac_NV"] = site_stats["n_matches"] / float(n_nv)
    # Now attach multiplicity estimate
    site_stats = attach_equiv_multiplicity(site_stats, c13_abundance=0.011)
    plot_multiplicity_hist(site_stats)
    plot_distance_vs_Nequiv(site_stats)
    plot_kappa_vs_distance(site_stats)
    plot_sites_3d_multiplicity(site_stats)

    # ------------------------------------------------------------------
    # 3) Text summary: which sites are most popular?
    # ------------------------------------------------------------------
    topN = 10  # how many top sites to print
    top_sites = site_stats.sort_values("n_matches", ascending=False).head(topN)

    print(
        "\nTop sites by number of NVs matched (for symmetry / multiplicity analysis):"
    )
    cols_to_show = [
        "orientation",
        "site_index",
        "distance_A",
        "x_A",
        "y_A",
        "z_A",
        "n_matches",
        "frac_NV",
        "kappa_mean",
    ]
    print(
        top_sites[cols_to_show].to_string(
            index=False, float_format=lambda x: f"{x:7.3f}"
        )
    )

    # ------------------------------------------------------------------
    # 4) Histogram of "how many sites had N matches?"
    # ------------------------------------------------------------------
    print("\nHistogram of site multiplicity (how many sites have N matches?):")
    mult_hist = site_stats["n_matches"].value_counts().sort_index()
    for n_match, count_sites in mult_hist.items():
        frac_sites = count_sites / n_unique_sites
        print(
            f"  n_matches = {n_match:2d}: {count_sites:4d} sites  "
            f"({100*frac_sites:5.2f}% of all unique sites)"
        )

    # ------------------------------------------------------------------
    # 2) Build a long-form branch table (one row per (NV, branch))
    # ------------------------------------------------------------------
    # Experimental pairs, orientation-aware
    plot_branch_pairs(
        matches["f1_fit_kHz"].to_numpy(),
        matches["f0_fit_kHz"].to_numpy(),
        title=f"{title_prefix}: experimental (f0, f1) pairs",
        exp_freqs=True,
        orientation=matches["orientation"].to_numpy(),
        ori_to_str=ori_to_str,
    )

    plot_branch_correlation_by_orientation(
        f1_kHz=matches["f1_fit_kHz"].to_numpy(),
        f0_kHz=matches["f0_fit_kHz"].to_numpy(),
        orientation=matches["orientation"].to_numpy(),
        title="ESEEM branch correlation by NV orientation (exp)",
        f_range_kHz=(10, 6000),
        filter_to_range=True,
        x_label="f0 (kHz)",
        y_label="f1 (kHz)",
    )

    plot_branch_pairs(
        matches["f_minus_kHz"].to_numpy(),
        matches["f_plus_kHz"].to_numpy(),
        title=f"{title_prefix}: matched theory (f_minus, f_plus) pairs",
        exp_freqs=False,
        orientation=matches["orientation"].to_numpy(),
        ori_to_str=ori_to_str,
    )

    plt.show()

    # ------------------------------------------------------------------
    # 3) Pick a single "matched" frequency per NV:
    #    use the branch with the smaller |err|
    # ------------------------------------------------------------------
    err0 = matches["err_f0_kHz"].to_numpy()
    err1 = matches["err_f1_kHz"].to_numpy()

    use_f0 = np.abs(err0) <= np.abs(err1)

    matches["branch_best"] = np.where(use_f0, "f0", "f1")
    matches["f_match_kHz"] = np.where(
        use_f0, matches["f_minus_kHz"], matches["f_plus_kHz"]
    )
    matches["err_match_kHz"] = np.where(use_f0, err0, err1)

    # orientation label
    matches["ori_label"] = matches["orientation"].apply(ori_to_str)

    # Decide markers: first orientation → circle, second → square
    unique_oris = list(matches["orientation"].dropna().unique())

    # optionally make ordering deterministic
    def _ori_key(o):
        try:
            return tuple(int(v) for v in o)
        except Exception:
            return tuple(o) if isinstance(o, (list, tuple)) else (o,)

    unique_oris.sort(key=_ori_key)

    marker_map = {}
    for idx, ori_val in enumerate(unique_oris):
        marker_map[ori_val] = "o" if idx == 0 else "s"  # 1st: circle, 2nd: square

    # Scatter: distance vs matched frequency (one point per NV), by orientation
    fig, ax = plt.subplots(figsize=(7, 5))

    sc = None
    for ori_val, sub in matches.groupby("orientation"):
        ori_lab = ori_to_str(ori_val)
        marker = marker_map.get(ori_val, "o")  # default circle if something weird

        sc = ax.scatter(
            sub["distance_A"],
            sub["f_match_kHz"],
            s=20,
            alpha=0.7,
            c=np.abs(sub["err_match_kHz"]),
            marker=marker,
            label=f"{ori_lab}",
        )

    ax.set_xlabel("Distance NV–13C (Å)")
    ax.set_ylabel("Matched frequency (kHz)")
    ax.set_title(
        f"{title_prefix}: best-matching ESEEM line vs distance (by orientation)",
        fontsize=15,
    )
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("frequency error |Δf| (kHz)", fontsize=15)

    ax.legend(title="Orientation", fontsize=8)

    # Columns that define a unique 13C lattice site
    site_key = ["orientation", "site_index", "x_A", "y_A", "z_A", "distance_A"]

    # 1) Group matches by site and count how many NVs matched each
    site_stats = matches.groupby(site_key, as_index=False).agg(
        n_matches=("nv_label", "count"),  # how many NVs matched this site
        kappa_mean=("kappa", "mean"),  # optional diagnostic
    )

    n_total_matches = len(matches)
    n_unique_sites = len(site_stats)

    print(f"Total NV–site matches: {n_total_matches}")
    print(f"Unique 13C sites:      {n_unique_sites}")

    # ------------------------------------------------------------------
    # 4) κ vs distance (unique sites)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sites["distance_A"], sites["kappa"], s=15, alpha=0.7)

    ax.set_xlabel("Distance NV–13C (Å)")
    ax.set_ylabel("κ (amplitude weight)")
    ax.set_title(f"{title_prefix}: κ vs distance", fontsize=15)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    plot_freq_vs_kappa(matches, title_prefix=title_prefix)

    return site_stats


def simulate_branch_pairs_like_exp(
    catalog,
    matches_df,
    c13_abundance=0.011,
    rng_seed=0,
    freq_minus_col="f_minus_Hz",
    freq_plus_col="f_plus_Hz",
    f_band_kHz=None,  # NEW: optional (f_min, f_max) in kHz
):
    """
    Simulate one ESEEM (f0, f1) pair per NV, matching the experimental
    NV count and per-NV orientation, with an optional frequency band
    and returning site information.

    Parameters
    ----------
    catalog : list of dicts or DataFrame
        Must contain columns:
          - 'orientation' (list or tuple, e.g. [-1,1,1])
          - freq_minus_col (Hz)
          - freq_plus_col  (Hz)
          - optional: 'kappa', 'site_index', 'x_A', 'y_A', 'z_A'
    matches_df : DataFrame
        Experimental matches, must contain:
          - 'nv_label'
          - 'orientation' (consistent with catalog orientation)
    c13_abundance : float
        13C fraction (natural ≈ 0.011).
    rng_seed : int
        Seed for reproducibility.
    f_band_kHz : tuple or None
        If not None, only consider sites whose f_minus or f_plus
        (in kHz) falls in [f_min, f_max] for selection.

    Returns
    -------
    f0_sim_kHz, f1_sim_kHz : (N_NV,)
    ori_list               : list of orientation tuples (length N_NV)
    site_index_sim         : (N_NV,)
    x_sim_A, y_sim_A, z_sim_A : (N_NV,)
    """
    # --- ensure catalog is a DataFrame ---
    if isinstance(catalog, list):
        df_full = pd.DataFrame(catalog)
    else:
        df_full = catalog.copy()

    for col in (freq_minus_col, freq_plus_col, "orientation"):
        if col not in df_full.columns:
            raise KeyError(f"Column '{col}' not found in catalog.")

    # normalize orientation in catalog to tuples
    def _norm_ori(o):
        if o is None:
            return None
        arr = np.asarray(o)
        if arr.size == 0:
            return None
        flat = arr.ravel()
        return tuple(int(v) for v in flat)

    df_full["ori_tuple"] = df_full["orientation"].apply(_norm_ori)

    # get unique NVs and their orientations from experiment
    nv_info = (
        matches_df[["nv_label", "orientation"]]
        .drop_duplicates("nv_label")
        .copy()
        .sort_values("nv_label")
    )
    # normalize NV orientations to tuples too
    nv_info["ori_tuple"] = nv_info["orientation"].apply(_norm_ori)
    nv_labels_sim = nv_info["nv_label"].to_numpy(int)

    N = len(nv_info)

    f0_sim_kHz = np.full(N, np.nan, float)
    f1_sim_kHz = np.full(N, np.nan, float)
    site_index_sim = np.full(N, np.nan, float)
    x_sim_A = np.full(N, np.nan, float)
    y_sim_A = np.full(N, np.nan, float)
    z_sim_A = np.full(N, np.nan, float)
    kappa_sim = np.full(N, np.nan, float)
    ori_list = []

    rng = np.random.default_rng(rng_seed)

    # loop over NVs, respect their experimental orientation
    for idx, row in nv_info.iterrows():
        i = nv_info.index.get_loc(idx)  # index 0..N-1
        ori_nv = row["ori_tuple"]
        ori_list.append(ori_nv)

        if ori_nv is None:
            continue

        # all sites in catalog with this orientation
        sub = df_full[df_full["ori_tuple"] == ori_nv]
        if sub.empty:
            continue

        M = len(sub)
        # random occupancy for this NV
        occ = rng.random(M) < c13_abundance
        if not np.any(occ):
            continue  # no occupied sites

        sub_occ = sub[occ].reset_index(drop=True)

        # --- apply frequency band, if requested ---
        if f_band_kHz is not None:
            f_min, f_max = f_band_kHz

            f_minus_kHz_all = sub_occ[freq_minus_col].to_numpy(float) * 1e-3
            f_plus_kHz_all = sub_occ[freq_plus_col].to_numpy(float) * 1e-3

            in_band = ((f_minus_kHz_all >= f_min) & (f_minus_kHz_all <= f_max)) & (
                (f_plus_kHz_all >= f_min) & (f_plus_kHz_all <= f_max)
            )

            if not np.any(in_band):
                # no occupied sites in this band → leave this NV as NaN
                continue

            sub_occ = sub_occ[in_band].reset_index(drop=True)

        # choose "dominant" site:
        # If kappa available, pick largest |kappa|; else choose random
        if "kappa" in sub_occ.columns and sub_occ["kappa"].notna().any():
            kappa_vals = sub_occ["kappa"].to_numpy(float)
            j = np.nanargmax(np.abs(kappa_vals))
        else:
            j = rng.integers(0, len(sub_occ))

        rec = sub_occ.iloc[j]

        # get frequencies in Hz, convert to kHz
        f_minus_kHz = float(rec[freq_minus_col]) * 1e-3
        f_plus_kHz = float(rec[freq_plus_col]) * 1e-3

        # assign in some consistent way; here we keep minus as f0, plus as f1
        f0_sim_kHz[i] = f_minus_kHz
        f1_sim_kHz[i] = f_plus_kHz

        # site info (if present)
        site_index_sim[i] = rec.get("site_index", np.nan)
        x_sim_A[i] = rec.get("x_A", np.nan)
        y_sim_A[i] = rec.get("y_A", np.nan)
        z_sim_A[i] = rec.get("z_A", np.nan)

        # kappa for that chosen site
        kappa_sim[i] = rec.get("kappa", np.nan)

    return (
        f0_sim_kHz,
        f1_sim_kHz,
        nv_labels_sim,
        np.array(ori_list, dtype=object),
        site_index_sim,
        x_sim_A,
        y_sim_A,
        z_sim_A,
        kappa_sim,
    )


def summarize_simulated_sites(
    matches_df,
    ori_list,
    site_index_sim,
    x_sim_A,
    y_sim_A,
    z_sim_A,
):
    """
    Build a per-site and per-NV summary of the simulated sites.

    Returns
    -------
    sim_sites : DataFrame
        One row per NV that got a site, columns:
          nv_label, orientation, site_index, x_A, y_A, z_A, distance_A,
          n_matches_sim (how many NVs picked this site),
          is_repeated_sim (True if this site was picked by >= 2 NVs).

    site_stats_sim : DataFrame
        One row per unique site, columns:
          orientation, site_index, x_A, y_A, z_A, distance_A,
          n_matches_sim
    """
    # get unique NVs + orientations in the same order as sim arrays
    nv_info = (
        matches_df[["nv_label", "orientation"]]
        .drop_duplicates("nv_label")
        .sort_values("nv_label")
        .reset_index(drop=True)
    )

    sim_sites = pd.DataFrame(
        {
            "nv_label": nv_info["nv_label"].to_numpy(),
            "orientation": nv_info["orientation"].to_numpy(),
            "site_index": site_index_sim,
            "x_A": x_sim_A,
            "y_A": y_sim_A,
            "z_A": z_sim_A,
        }
    )

    # drop NVs that didn't get a site
    sim_sites = sim_sites[np.isfinite(sim_sites["site_index"])].copy()

    # distance
    sim_sites["distance_A"] = np.sqrt(
        sim_sites["x_A"] ** 2 + sim_sites["y_A"] ** 2 + sim_sites["z_A"] ** 2
    )

    # define site key (same as in exp analysis)
    site_key = ["orientation", "site_index", "x_A", "y_A", "z_A", "distance_A"]

    # per-site multiplicity in the simulation
    site_stats_sim = sim_sites.groupby(site_key, as_index=False).agg(
        n_matches_sim=("nv_label", "count")
    )

    # merge back to NV-level to know, for each NV, how popular its site is
    sim_sites = sim_sites.merge(
        site_stats_sim,
        on=site_key,
        how="left",
    )

    # mark repeated sites
    sim_sites["is_repeated_sim"] = sim_sites["n_matches_sim"] >= 2

    return sim_sites, site_stats_sim


def plot_simualted(
    matches_df,
    f0_sim_kHz,
    f1_sim_kHz,
    ori_sim,
    site_index_sim,
    x_sim_A,
    y_sim_A,
    z_sim_A,
):
    # 2) Positions of simulated sites that actually contributed (non-NaN)
    mask_valid = np.isfinite(f0_sim_kHz) & np.isfinite(f1_sim_kHz)

    x_band = x_sim_A[mask_valid]
    y_band = y_sim_A[mask_valid]
    z_band = z_sim_A[mask_valid]
    ori_band_raw = ori_sim[mask_valid]  # orientations for valid NVs

    # Turn orientations into nice string labels, e.g. "(+1-1+1)"
    ori_labels = np.array([ori_to_str(o) for o in ori_band_raw], dtype=object)

    # Unique orientation labels (e.g. two of them)
    unique_labels = sorted(set(ori_labels))

    # 1) Branch-pair plot (same style as experiment)
    plot_branch_pairs(
        f1_sim_kHz,
        f0_sim_kHz,
        title=f"204NVs: simulated (f0, f1) pairs ({band[0]:.0f}–{band[1]:.0f}) kHz",
        exp_freqs=True,
        orientation=ori_sim,
        ori_to_str=ori_to_str,
    )
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    markers = ["o", "^", "s", "D"]  # enough if more show up

    for idx, lab in enumerate(unique_labels):
        if lab == "ori=None":
            continue  # skip any bad / missing orientation entries

        m = markers[idx % len(markers)]
        mask_ori = ori_labels == lab

        ax.scatter(
            x_band[mask_ori],
            y_band[mask_ori],
            z_band[mask_ori],
            s=15,
            alpha=0.9,
            marker=m,
            label=lab,  # label is already something like "(+1-1+1)"
        )

    # NV at origin
    ax.scatter(
        0,
        0,
        0,
        marker="*",
        s=60,
        edgecolor="k",
        linewidth=0.6,
        label="NV",
    )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(f"Simulated sites with lines in {band[0]:.0f}–{band[1]:.0f} kHz")
    ax.legend(title="Orientation", fontsize=11, title_fontsize=11)

    sim_sites, site_stats_sim = summarize_simulated_sites(
        matches_df,
        ori_sim,
        site_index_sim,
        x_sim_A,
        y_sim_A,
        z_sim_A,
    )

    # 3D plot of unique simulated sites, color = n_matches_sim, with colorbar
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        site_stats_sim["x_A"],
        site_stats_sim["y_A"],
        site_stats_sim["z_A"],
        c=site_stats_sim["n_matches_sim"],  # color = how many NVs picked this site
        s=10,
        alpha=0.9,
    )

    # NV at origin
    ax.scatter(
        0,
        0,
        0,
        marker="*",
        s=60,
        edgecolor="k",
        linewidth=0.6,
        label="NV",
    )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title(
        f"Simulated sites in {band[0]:.0f}–{band[1]:.0f} kHz\n" "color = n_matches_sim",
    )

    cbar = fig.colorbar(p, ax=ax, shrink=0.75)
    cbar.set_label("n_matches_sim (NVs per site)")

    ax.legend(loc="upper left")

    print("Simulated site multiplicity (n_matches_sim):")
    print(site_stats_sim["n_matches_sim"].value_counts().sort_index())


def extract_unified_fit_params(fit_file_stem: str) -> pd.DataFrame:
    d = dm.get_raw_data(file_stem=fit_file_stem)

    nv = np.asarray(d["nv_labels"], dtype=int)
    popts = d["popts"]
    red = np.asarray(d.get("red_chi2", [np.nan] * len(nv)), dtype=float)

    unified_keys = d.get("unified_keys", None)
    if not isinstance(unified_keys, list):
        raise KeyError(f"{fit_file_stem}: missing/invalid 'unified_keys'")

    # Known reduced schema when the fitter returns a shorter parameter vector.
    # (This is the most common compact ESEEM fit layout.)
    reduced6_keys = [
        "baseline",
        "comb_contrast",
        "T2_ms",
        "osc_amp",
        "osc_f0",
        "osc_f1",
    ]

    # Choose key list based on popt length
    schema_by_len = {
        len(unified_keys): unified_keys,
        6: reduced6_keys,
    }

    # Debug: count popt lengths
    lens = [len(p) for p in popts if p is not None]
    if lens:
        uniq = {L: lens.count(L) for L in sorted(set(lens))}
        print(f"[extract_unified_fit_params] {fit_file_stem}: popt lengths -> {uniq}")

    rows = []
    for nv_label, p, rchi in zip(nv, popts, red):
        if p is None:
            continue
        p = np.asarray(p, float)
        L = len(p)

        keys = schema_by_len.get(L, None)
        if keys is None:
            # Unknown schema length; keep row but with NaNs so merge works
            rows.append(
                dict(
                    nv_label=int(nv_label),
                    comb_contrast=np.nan,
                    osc_amp=np.nan,
                    red_chi2=float(rchi),
                )
            )
            continue

        idx = {k: i for i, k in enumerate(keys)}

        def getv(name):
            i = idx.get(name, None)
            return float(p[i]) if (i is not None and i < L) else np.nan

        rows.append(
            dict(
                nv_label=int(nv_label),
                comb_contrast=getv("comb_contrast"),
                osc_amp=getv("osc_amp"),
                red_chi2=float(rchi),
                osc_f0_cyc_per_us=getv("osc_f0"),
                osc_f1_cyc_per_us=getv("osc_f1"),
            )
        )

    return pd.DataFrame(rows).drop_duplicates(subset=["nv_label"])


def run_field_analysis(
    label: str,
    fit_file_stem: str,
    counts_file_stem: str,
    B_G: np.ndarray,
    catalog_json: str,
    distance_cutoff_A: float = 15.0,
) -> dict:
    """Run your full pipeline for one B field and return a compact summary."""
    # 1) Load catalog + hyperfine
    hf_df = load_hyperfine_table(distance_cutoff=distance_cutoff_A)
    catalog_records = load_catalog(catalog_json)

    # 2) Extract fit summary for this field
    fit_summary = freqs_from_popts_exact(file_stem=fit_file_stem)
    nv = np.asarray(fit_summary["nv"], int)
    f0_kHz = np.asarray(fit_summary["f0_kHz"], float)
    f1_kHz = np.asarray(fit_summary["f1_kHz"], float)
    fit_fail = np.asarray(fit_summary["fit_fail"], bool)
    nv_oris = np.asarray(fit_summary["orientations"], int)
    site_ids = np.asarray(fit_summary["site_ids"], int)

    mask = (
        np.isfinite(f0_kHz)
        & np.isfinite(f1_kHz)
        & (f0_kHz > 0)
        & (f1_kHz >= 0)
        & (~fit_fail)
    )

    nv_kept = nv[mask]
    f0_kept_kHz = f0_kHz[mask]
    f1_kept_kHz = f1_kHz[mask]
    site_ids_kept = site_ids[mask]
    nv_oris_kept = nv_oris[mask]

    # 3) Matching to catalog for this field
    matches_df = pairwise_match_from_site_ids_kHz(
        nv_labels=nv_kept,
        f0_kHz=f0_kept_kHz,
        f1_kHz=f1_kept_kHz,
        site_ids=site_ids_kept,
        nv_orientations=nv_oris_kept,
        records=catalog_records,
    )

    # -------------------- 3b) QUALITY CUTS --------------------
    # attach comb_contrast and osc_amp from the same fit file
    uni_df = extract_unified_fit_params(fit_file_stem)

    matches_df = matches_df.merge(
        uni_df,
        on="nv_label",
        how="left",
    )

    # normalized ESEEM amplitude
    matches_df["osc_amp_norm"] = matches_df["osc_amp"] / matches_df["comb_contrast"]

    # define a "good physics" mask
    mask_good = (
        np.isfinite(matches_df["kappa"])
        & np.isfinite(matches_df["osc_amp_norm"])
        & np.isfinite(matches_df["comb_contrast"])
        & (matches_df["comb_contrast"] > 0.02)  # non-trivial echo
        & (np.abs(matches_df["osc_amp_norm"]) < 3.0)  # kill ratios like 40, 100
        & (matches_df["red_chi2"] < 3.0)  # decent fit
    )

    n_before = matches_df["nv_label"].nunique()
    matches_df = matches_df[mask_good].copy()
    n_after = matches_df["nv_label"].nunique()
    print(
        f"[{label}] kept {n_after} / {n_before} matched NVs after "
        f"amp/chi² quality cuts"
    )
    # -----------------------------------------------------------

    # 4) Full site multiplicity analysis (now only with good NVs)
    orbit_df = find_c3v_orbits_from_nv2(
        hyperfine_path=HYPERFINE_PATH,
        r_max_A=22.0,
        tol_r_A=0.02,
        tol_dir=5e-2,
    )
    site_stats_full, orbit_stats = build_site_multiplicity_with_theory(
        matches_df=matches_df,
        orbit_df=orbit_df,
        p13=0.011,
    )

    # Attach metadata so you can stack later
    B_mag_G = float(np.linalg.norm(B_G))
    matches_df["field_label"] = label
    matches_df["B_mag_G"] = B_mag_G

    matches_df["Bx_G"] = float(B_G[0])
    matches_df["By_G"] = float(B_G[1])
    matches_df["Bz_G"] = float(B_G[2])

    site_stats_full = site_stats_full.copy()
    site_stats_full["field_label"] = label
    site_stats_full["B_mag_G"] = B_mag_G

    orbit_stats = orbit_stats.copy()
    orbit_stats["field_label"] = label
    orbit_stats["B_mag_G"] = B_mag_G

    # (Optional) frac matched
    frac_matched = matches_df["nv_label"].nunique() / nv_kept.size
    uni_df = extract_unified_fit_params(fit_file_stem)

    return dict(
        label=label,
        B_G=B_G,
        B_mag_G=B_mag_G,
        matches_df=matches_df,
        site_stats=site_stats_full,
        orbit_stats=orbit_stats,
        frac_matched=frac_matched,
        n_nv_total=int(nv_kept.size),
        unified_df=uni_df,
    )


def plot_site_f_vs_B(all_matches: pd.DataFrame, site_list=None):
    """
    For each (orientation, site_index), plot matched f_minus/f_plus vs B_mag_G.
    Optionally restrict to a few interesting sites.
    """
    df = all_matches.copy()
    # key that identifies a physical 13C site
    df["site_key"] = list(zip(df["orientation"], df["site_index"]))

    if site_list is not None:
        df = df[df["site_key"].isin(site_list)]

    # keep only sites seen in more than one field
    counts = df.groupby("site_key")["field_label"].nunique()
    multi = counts[counts > 2].index
    df = df[df["site_key"].isin(multi)]

    for site_key, sub in df.groupby("site_key"):
        ori, sid = site_key
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(
            sub["B_mag_G"],
            sub["f_minus_kHz"],
            yerr=np.abs(sub["err_f0_kHz"]),  # or err on the branch you used
            fmt="o-",
            label="f_minus",
        )
        ax.errorbar(
            sub["B_mag_G"],
            sub["f_plus_kHz"],
            yerr=np.abs(sub["err_f1_kHz"]),
            fmt="s-",
            label="f_plus",
        )
        ax.set_xlabel("|B| (G)")
        ax.set_ylabel("frequency (kHz)")
        ax.set_title(f"Site {sid}, ori={ori}")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()


def plot_site_fplus_vs_fminus_and_angles(
    all_matches: pd.DataFrame,
    site_list=None,
    min_fields: int = 4,
    annotate: bool = True,
    use_one_point_per_field: bool = True,  # True: one point per field (recommended)
    ax_equal_f: bool = False,  # f+ vs f- equal aspect
):
    """
    For each (orientation, site_index) that appears in >=min_fields:

      Panel A:  f_plus vs f_minus (with xerr/yerr if available)
      Panel B:  B direction scatter in (phi, theta) for the same points

    Requires:
      field_label, orientation, site_index
      f_minus_kHz, f_plus_kHz
      err_f0_kHz, err_f1_kHz  (optional but recommended)
      B_phi_deg, B_theta_deg  (compute with attach_Bhat_theta_phi)
    """

    df = all_matches.copy()
    df["site_key"] = list(zip(df["orientation"], df["site_index"]))

    if site_list is not None:
        df = df[df["site_key"].isin(site_list)]

    # keep only sites seen in >= min_fields
    n_by_site = df.groupby("site_key")["field_label"].nunique()
    keep_sites = n_by_site[n_by_site >= min_fields].index
    df = df[df["site_key"].isin(keep_sites)]

    if len(df) == 0:
        print("No sites satisfy the min_fields/site_list filters.")
        return

    need_cols = ["B_phi_deg", "B_theta_deg", "f_minus_kHz", "f_plus_kHz", "field_label"]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(
                f"Missing column '{c}'. Did you run attach_Bhat_theta_phi()?"
            )

    for site_key, sub in df.groupby("site_key"):
        ori, sid = site_key
        sub = sub.copy()

        # keep one representative per field (since direction is field-level anyway)
        if use_one_point_per_field:
            sub = sub.sort_values(
                "err_pair_kHz" if "err_pair_kHz" in sub.columns else "field_label"
            )
            sub = sub.drop_duplicates("field_label", keep="first")

        # errors (optional)
        xerr = np.abs(sub["err_f0_kHz"]) if "err_f0_kHz" in sub.columns else None
        yerr = np.abs(sub["err_f1_kHz"]) if "err_f1_kHz" in sub.columns else None

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.5))
        # ---------------- Panel A: f+ vs f- ----------------
        for lab, g in sub.groupby("field_label"):
            ax0.errorbar(
                g["f_minus_kHz"],
                g["f_plus_kHz"],
                xerr=np.abs(g["err_f0_kHz"]) if "err_f0_kHz" in g else None,
                yerr=np.abs(g["err_f1_kHz"]) if "err_f1_kHz" in g else None,
                fmt="o",
                capsize=2,
                label=lab,
            )

        ax0.set_xlabel(r"$f_-$ (kHz)")
        ax0.set_ylabel(r"$f_+$ (kHz)")
        ax0.set_title(r"$f_+$ vs $f_-$")
        ax0.grid(True, alpha=0.25)

        # Optional: equal aspect helps you see “bending” vs monotonic separation
        if ax_equal_f:
            ax0.set_aspect("equal", adjustable="box")

        # ---------------- Panel B: direction (phi, theta) ----------------
        for lab, g in sub.groupby("field_label"):
            ax1.scatter(g["B_phi_deg"], g["B_theta_deg"], label=lab)

        if annotate:
            for _, r in sub.iterrows():
                ax1.text(
                    r["B_phi_deg"] + 2,
                    r["B_theta_deg"] + 2,
                    str(r["field_label"]),
                    fontsize=9,
                )

        ax1.set_xlabel(r"azimuth $\phi$ [deg]")
        ax1.set_ylabel(r"polar $\theta$ [deg]")
        ax1.set_title(r"$\hat B$ direction in $(\phi,\theta)$")
        # ax1.set_xlim(-180, 180)
        # ax1.set_ylim(0, 180)
        ax1.grid(True, alpha=0.25)

        # legend (shared meaning across both panels)
        ax0.legend(fontsize=9, loc="best")
        # (optional) ax1.legend(...) if you want it twice

        fig.suptitle(f"Site {sid}, ori={ori}", y=0.98, fontsize=13)
        # fig.subplots_adjust(top=0.88, wspace=0.30)


def compare_multiplicity_across_fields(all_site_stats):
    # site key without field
    key = ["orientation", "site_index"]

    # how many fields does each site show up in?
    field_counts = all_site_stats.groupby(key)["field_label"].nunique().reset_index()
    field_counts.rename(columns={"field_label": "n_fields_seen"}, inplace=True)

    # average n_matches per field for sites that appear in multiple fields
    agg = all_site_stats.groupby(key + ["field_label"], as_index=False).agg(
        n_matches=("n_matches", "mean")
    )

    print("Sites seen in ≥2 fields:")
    print(field_counts[field_counts["n_fields_seen"] >= 2].sort_values("n_fields_seen"))

    # Simple example: histogram of n_matches per field
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, sub in all_site_stats.groupby("field_label"):
        ax.hist(
            sub["n_matches"],
            bins=np.arange(0.5, sub["n_matches"].max() + 1.5),
            histtype="step",
            label=label,
        )
    ax.set_xlabel("n_matches (NVs per site)")
    ax.set_ylabel("# of sites")
    ax.set_title("Site multiplicity distribution vs field")
    ax.legend()
    return fig, ax


def compare_NV_assignments(all_matches):
    # one row per NV per field
    df = all_matches[["nv_label", "field_label", "site_index", "err_pair_kHz"]].copy()
    # keep only NVs that appear in more than one field
    multi_nv = df.groupby("nv_label")["field_label"].nunique()
    multi_nv = multi_nv[multi_nv > 1].index
    df = df[df["nv_label"].isin(multi_nv)]

    # how often does a given NV change site assignment?
    site_spread = df.groupby("nv_label")["site_index"].nunique().reset_index()
    site_spread.rename(columns={"site_index": "n_distinct_sites"}, inplace=True)

    print("NVs with changing site assignment across fields:")
    print(site_spread[site_spread["n_distinct_sites"] > 1].head(20))


def canonicalize_orientation_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["orientation"] = out["orientation"].apply(
        lambda o: tuple(np.asarray(o, int).tolist())
    )
    out["site_index"] = out["site_index"].astype(int)
    return out


def attach_field_vectors(all_matches: pd.DataFrame) -> pd.DataFrame:
    df = all_matches.copy()
    B = df[["Bx_G", "By_G", "Bz_G"]].to_numpy(float)
    Bmag = np.linalg.norm(B, axis=1)
    df["B_mag_G"] = Bmag

    Bhat = B / Bmag[:, None]
    df["Bhat_x"] = Bhat[:, 0]
    df["Bhat_y"] = Bhat[:, 1]
    df["Bhat_z"] = Bhat[:, 2]

    # lon/lat (deg) for easy direction plotting
    df["B_lon_deg"] = np.degrees(np.arctan2(df["Bhat_y"], df["Bhat_x"]))  # [-180,180]
    df["B_lat_deg"] = np.degrees(np.arcsin(np.clip(df["Bhat_z"], -1, 1)))  # [-90,90]
    return df


def attach_lab_angles(df: pd.DataFrame, phi_range="0_360") -> pd.DataFrame:
    out = df.copy()

    if {"Bhat_x", "Bhat_y", "Bhat_z"}.issubset(out.columns):
        bx = out["Bhat_x"].to_numpy(float)
        by = out["Bhat_y"].to_numpy(float)
        bz = out["Bhat_z"].to_numpy(float)
    else:
        B = out[["Bx_G", "By_G", "Bz_G"]].to_numpy(float)
        Bh = B / np.linalg.norm(B, axis=1, keepdims=True)
        bx, by, bz = Bh[:, 0], Bh[:, 1], Bh[:, 2]

    theta = np.degrees(np.arccos(np.clip(bz, -1.0, 1.0)))  # 0..180 (polar from +Z)
    phi = np.degrees(np.arctan2(by, bx))  # -180..180 (azimuth around +Z)

    if phi_range == "0_360":
        phi = np.mod(phi, 360.0)
    elif phi_range == "-180_180":
        phi = ((phi + 180.0) % 360.0) - 180.0

    out["B_theta_deg"] = theta
    out["B_phi_deg"] = phi
    return out


if __name__ == "__main__":
    kpl.init_kplotlib()
    # -------------------- Main analysis per field --------------------
    field_cfgs = [
        dict(
            label="49G",
            fit_file_stem="2025_11_19-14_19_23-sample_204nv_s1-fcc605",
            counts_file_stem="2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c",
            B_G=np.array([-46.27557688, -17.16599864, -5.70139829]),
            catalog_json="analysis/spin_echo_work/essem_freq_kappa_catalog_22A_49G.json",
        ),
        dict(
            label="59G",
            fit_file_stem="2025_12_05-07_51_13-sample_204nv_s1-4cf818",
            counts_file_stem="2025_12_04-19_50_15-johnson_204nv_s9-2c83ab",
            B_G=np.array([-41.57848995, -32.77145194, -27.5799348]),
            catalog_json="analysis/spin_echo_work/essem_freq_kappa_catalog_22A_59G.json",
        ),
        dict(
            label="62G",
            # fit_file_stem="2025_12_25-04_45_34-sample_204nv_s1-752556",
            # counts_file_stem="2025_12_24-19_56_03-johnson_204nv_s6-ff5e17",
            fit_file_stem="2026_01_07-22_29_55-sample_204nv_s1-c9580a",
            counts_file_stem="2026_01_07-17_30_14-johnson_204nv_s12-06bcdd",
            B_G=np.array([-48.67047318, -32.07615947, 22.49657427]),
            catalog_json="analysis/spin_echo_work/essem_freq_kappa_catalog_22A_62G.json",
        ),
        dict(
            label="65G",
            # fit_file_stem="2025_11_30-04_35_04-sample_204nv_s1-d278ee",
            # counts_file_stem="2025_11_28-16_39_32-johnson_204nv_s6-902522",
            fit_file_stem="2026_01_07-23_43_15-sample_204nv_s1-27e4dc",
            counts_file_stem="2026_01_07-17_48_21-johnson_204nv_s10-34f8b7",
            B_G=np.array([-31.61263115, -56.58135644, -6.5512002]),
            catalog_json="analysis/spin_echo_work/essem_freq_kappa_catalog_22A_65G.json",
        ),
    ]

    results = [run_field_analysis(**cfg) for cfg in field_cfgs]

    all_matches = pd.concat([r["matches_df"] for r in results], ignore_index=True)
    all_site_stats = pd.concat([r["site_stats"] for r in results], ignore_index=True)

    # plot_site_f_vs_B(all_matches)
    # plot_site_f_vs_B_and_direction(all_matches, min_fields=4, ref_field_label="49G")
    # plot_site_f_vs_B_and_direction(all_matches, site_list=[28, 72],  min_fields=4, ref_field_label="49G")

    all_matches = attach_field_vectors(all_matches)  # gives Bhat_x/y/z
    all_matches = attach_lab_angles(all_matches)  # adds B_phi_deg, B_theta_deg

    plot_site_fplus_vs_fminus_and_angles(
        all_matches,
        min_fields=4,
        use_one_point_per_field=True,
    )
    plt.show()

    # compare_multiplicity_across_fields(all_site_stats)
    # compare_NV_assignments(all_matches)
    # plot_T2_vs_field(all_matches)
    # plot_T2_vs_distance(all_matches)
    # pick a few sites with large n_matches for f(B) tracks
    plt.show(block=True)
    # sys.exit()

    all_matches = pd.concat(
        [
            res["matches_df"].assign(
                field_label=cfg["label"],
                Bx_G=cfg["B_G"][0],
                By_G=cfg["B_G"][1],
                Bz_G=cfg["B_G"][2],
            )
            for cfg, res in zip(field_cfgs, results)
        ],
        ignore_index=True,
    )
    wide_both = compare_two_fields(
        all_matches,
        field_labels=[
            "49G",
            "59G",
            "62G",
            "65G",
        ],  # explicit, or leave None to auto-detect the two
        title_prefix="204 NVs",
    )

    all_site_stats2 = canonicalize_orientation_col(all_site_stats)

    field_labels = sorted(all_site_stats2["field_label"].unique())
    n_fields = len(field_labels)
    print("Fields:", field_labels, " n_fields=", n_fields)

    site_counts = (
        all_site_stats2.groupby(["orientation", "site_index"])["field_label"]
        .nunique()
        .sort_values(ascending=False)
    )

    sites_all = site_counts[site_counts == n_fields].index.tolist()
    print(f"# sites seen in ALL {n_fields} fields:", len(sites_all))
    print("Example (first 10):", sites_all[:10])

    rank_df = (
        all_site_stats2[
            all_site_stats2.set_index(["orientation", "site_index"]).index.isin(
                sites_all
            )
        ]
        .groupby(["orientation", "site_index"], as_index=False)
        .agg(n_matches_sum=("n_matches", "sum"), n_matches_mean=("n_matches", "mean"))
        .sort_values("n_matches_sum", ascending=False)
    )
    print(rank_df.head(20))

    B_by_field = {r["label"]: np.asarray(r["B_G"], float) for r in results}
    for lab, B in B_by_field.items():
        print(lab, " |B|=", np.linalg.norm(B), " Bhat=", B / np.linalg.norm(B))

    site_key0 = tuple(rank_df.iloc[0][["orientation", "site_index"]])  # top ranked site
    print("Top ranked site seen in all fields:", site_key0)

    ori, sid = site_key0
    overlay_points = []
    for lab, Bvec_G in B_by_field.items():
        b = np.asarray(Bvec_G, float)
        overlay_points.append({"label": lab, "Bhat": b / np.linalg.norm(b)})

    # angle_map = compute_full_angle_map_lab_spherical_for_site(
    #     hyperfine_path=HYPERFINE_PATH,
    #     orientation=ori,
    #     site_index=sid,
    #     Bmag_G=62.0,  # pick a representative magnitude (or do one per field)
    #     n_theta=91,
    #     n_phi=181,
    #     ms=-1,
    #     phi_deg_nv=0.0,
    #     phi_range="0_360",
    # )

    # plot_full_angle_map_theta_phi_lab(
    #     angle_map,
    #     title_prefix=f"Site {sid}, ori={ori}",
    #     contour=True,
    #     add_kappa_contours=True,
    #     overlay_points=overlay_points,
    # )

    # Example inputs (replace with your real ones)
    phi_deg_nv_used = 0.0  # or 120.0 — MUST match what your catalog/fitting used
    sid = 72
    ori = (-1, 1, 1)  # see note below on how to get this from your tables

    cfg = field_cfgs[0]  # 49G
    Bvec_G = cfg["B_G"]
    Bmag_G = float(np.linalg.norm(Bvec_G))

    # validate_site72_direct_vs_labmap(
    #     hyperfine_path=HYPERFINE_PATH,
    #     orientation=ori,  # must match the site72 orientation used in fit/catalog
    #     site_index=72,
    #     Bvec_G=Bvec_G,
    #     Bmag_G=Bmag_G,
    #     ms=-1,
    #     phi_deg_nv=0.0,  # must match fit/catalog
    #     phi_range="0_360",
    #     n_theta=181,
    #     n_phi=360,
    # )

    plt.show(block=True)
    # sys.exit()
    # --- Magnetic field (crystal axes) ---
    # B_G = [-46.27557688 - 17.16599864 - 5.70139829]
    # B_G_mag = 49.685072884712
    # B_hat = [-0.93137786 - 0.34549609 - 0.11475073]
    # fit_file_stem    = "2025_11_13-06_28_22-sample_204nv_s1-e85aa7"   # where popts & freqs live
    # fit_file_stem  = "2025_11_14-03_05_30-sample_204nv_s1-e85aa7" # 200 freqs freeze
    # fit_file_stem  = "2025_11_14-18_28_58-sample_204nv_s1-e85aa7" # 600 freqs freeze
    fit_file_stem = "2025_11_17-09_49_42-sample_204nv_s1-fcc605"  # site encoded, all freqs (nysq band)
    # fit_file_stem = (
    #     "2025_11_19-14_19_23-sample_204nv_s1-fcc605"  # site encoded, 1500 freqs pairs (1khz-6Mhz)
    # )
    counts_file_stem = (
        "2025_11_11-01_15_45-johnson_204nv_s6-6d8f5c"  # merged dataset2+3 counts
    )
    catalog_json = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_49G.json"

    # --- Magnetic field (crystal axes) ---
    B_G = [-31.61263115 - 56.58135644 - 6.5512002]
    B_G = 65.143891267575
    B_G = [-0.48527391 - 0.86855967 - 0.10056507]
    # fit_file_stem = "2025_11_30-04_35_04-sample_204nv_s1-d278ee"  # site encoded, all freqs (nysq band)
    # counts_file_stem = "2025_11_28-16_39_32-johnson_204nv_s6-902522"
    fit_file_stem = "2026_01_07-23_43_15-sample_204nv_s1-27e4dc"  # site encoded, all freqs (nysq band)
    counts_file_stem = "2026_01_07-17_48_21-johnson_204nv_s10-34f8b7"
    catalog_json = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_65G.json"

    # --- Magnetic field (crystal axes) ---"59G"
    # B_G = np.array([-41.57848995, -32.77145194, -27.5799348])
    # fit_file_stem = "2025_12_05-07_51_13-sample_204nv_s1-4cf818"
    # counts_file_stem = "2025_12_04-19_50_15-johnson_204nv_s9-2c83ab"
    # catalog_json = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_59G.json"

    # --- Magnetic field (crystal axes) ---
    # B_G = np.array([-48.67047318, -32.07615947, 22.49657427])
    # fit_file_stem = "2025_12_25-04_45_34-sample_204nv_s1-752556"
    # counts_file_stem = "2025_12_24-19_56_03-johnson_204nv_s6-ff5e17"

    # fit_file_stem = "2026_01_07-22_29_55-sample_204nv_s1-c9580a"
    # counts_file_stem = "2026_01_07-17_30_14-johnson_204nv_s12-06bcdd"

    # catalog_json = "analysis/spin_echo_work/essem_freq_kappa_catalog_22A_62G.json"

    ## ---- 2) global theory-vs-exp matching (use FIT file) ----##
    hf_df = load_hyperfine_table(distance_cutoff=15.0)  # or 15.0, etc.
    data = dm.get_raw_data(file_stem=fit_file_stem)
    fit_summary = freqs_from_popts_exact(file_stem=fit_file_stem)

    nv = np.asarray(fit_summary["nv"], int)
    T2_us = np.asarray(fit_summary["T2_us"], float)
    f0_kHz = np.asarray(fit_summary["f0_kHz"], float)
    f1_kHz = np.asarray(fit_summary["f1_kHz"], float)
    A_pick_kHz = np.asarray(fit_summary["A_pick_kHz"], float)
    chis = np.asarray(fit_summary["chis"], float)
    fit_fail = np.asarray(fit_summary["fit_fail"], bool)
    sT2_us = np.asarray(fit_summary["sT2_us"], float)
    sf0_kHz = np.asarray(fit_summary["sf0_kHz"], float)
    sf1_kHz = np.asarray(fit_summary["sf1_kHz"], float)
    sA_pick_kHz = np.asarray(fit_summary["sA_pick_kHz"], float)

    nv_oris = np.asarray(fit_summary["orientations"], int)
    site_ids = np.asarray(fit_summary["site_ids"], int)

    mask = (
        np.isfinite(f0_kHz)
        & np.isfinite(f1_kHz)
        & (f0_kHz > 0)
        & (f1_kHz >= 0)
        & (~np.array(fit_fail, dtype=bool))
    )

    nv_kept = nv[mask]
    f0_kept_kHz = f0_kHz[mask]
    f1_kept_kHz = f1_kHz[mask]
    site_ids_kept = site_ids[mask]
    nv_oris_kept = nv_oris[mask]
    if f0_kept_kHz.shape != f1_kept_kHz.shape:
        raise ValueError("f0_kHz and f1_kHz must have same shape.")
    if f0_kept_kHz.ndim != 1:
        raise ValueError("f0_kHz and f1_kHz must be 1D.")
    if site_ids_kept.shape[0] != f0_kept_kHz.shape[0]:
        raise ValueError("site_ids must have same length as f0_kHz/f1_kHz.")

    exp_pairs_with_labels = list(
        zip(
            nv_kept.tolist(),
            [(float(f0), float(f1)) for f0, f1 in zip(f0_kept_kHz, f1_kept_kHz)],
        )
    )

    catalog_records = load_catalog(catalog_json)
    matches_df = pairwise_match_from_site_ids_kHz(
        nv_labels=nv_kept,
        f0_kHz=f0_kept_kHz,
        f1_kHz=f1_kept_kHz,
        site_ids=site_ids_kept,
        nv_orientations=nv_oris_kept,
        records=catalog_records,
    )

    print(matches_df["err_pair_kHz"].describe())
    row = matches_df.loc[matches_df["nv_label"] == 137]
    print(row.T)

    print(matches_df.head())
    cat_df = pd.DataFrame(catalog_records)
    orientation = cat_df["orientation"].to_numpy()
    plot_branch_correlation_by_orientation(
        f0_kHz=cat_df["f_minus_Hz"].to_numpy() / 1e3,
        f1_kHz=cat_df["f_plus_Hz"].to_numpy() / 1e3,
        orientation=cat_df["orientation"].to_numpy(),
        # orientation=[[1, 1, 1]],
        title="Catalog branch correlation: f_- vs f_+",
        f_range_kHz=(10, 6000),
        filter_to_range=True,
        x_label="f_- (kHz)",
        y_label="f_+ (kHz)",
    )
    # site_stats = analyze_matched_c13_sites(matches_df, title_prefix="204 NVs")

    # ---- 3) echo trace and corresponding matched site ----

    # nv_list = [0, 1, 2, 137]  # whatever NVs you care about

    nv_list = [0, 1]  # whatever NVs you care about
    # nv_list = nv_kept
    figs = make_echo_plus_matched_site_plots_batch(
        counts_file_stem=counts_file_stem,
        fit_file_stem=fit_file_stem,
        matches_enriched=matches_df,  # from pairwise_match_from_site_ids_kHz
        hf_df=hf_df,  # <- only use matched site coordinates
        nv_labels=nv_list,
        use_half_time_as_tau=False,
    )
    plt.show(block=True)
    sys.exit()
    ## ---- 4) simulations template set by experiment ----##
    exp_f = matches_df["f_minus_kHz"].to_numpy(float)
    f_band_kHz = (np.nanmin(exp_f), np.nanmax(exp_f))
    n_nv = matches_df["nv_label"].nunique()
    band = f_band_kHz  # kHz
    # # band = (10, 1500)  # kHz

    (
        f0_sim_kHz,
        f1_sim_kHz,
        nv_labels_sim,
        ori_sim,
        site_index_sim,
        x_sim_A,
        y_sim_A,
        z_sim_A,
        kappa_sim,
    ) = simulate_branch_pairs_like_exp(
        catalog_records,
        matches_df=matches_df,
        c13_abundance=0.011,
        rng_seed=20,
        freq_minus_col="f_minus_Hz",
        freq_plus_col="f_plus_Hz",
        f_band_kHz=band,
    )
    # sys.exit()
    # ---- 5) Multiplicity Analsysis by both theory and experiment ----##
    orbit_df = find_c3v_orbits_from_nv2(
        hyperfine_path=HYPERFINE_PATH,
        r_max_A=22.0,  # or 15.0, to match your catalog cutoff
        tol_r_A=0.02,  # 0.02 Å is usually fine
        tol_dir=5e-2,  # ~0.05 in unit-vector norm (~few degrees)
    )
    print(orbit_df.head(20))
    # See the multiplicity stats
    print("\nMultiplicity histogram (theory):")
    print(orbit_df["n_equiv_theory"].value_counts().sort_index())

    # Experiment
    # site_stats_exp = build_site_multiplicity_with_theory(
    #     matches_df=matches_df,
    #     orbit_df=orbit_df,
    #     p13=0.011,  # natural abundance
    # )
    # multiplicity_plots(site_stats_exp, dataset_label="Experiment")
    # make_a_table(site_stats_exp, topN=15, dataset_label="Experiment")

    # print(
    #     site_stats_full.sort_values("n_matches", ascending=False)[cols_to_show]
    #     .head(15)
    #     .to_string(index=False)
    # )

    # # Simulation
    plot_simualted(
        f0_sim_kHz,
        f1_sim_kHz,
        ori_sim,
        site_index_sim,
        x_sim_A,
        y_sim_A,
        z_sim_A,
    )

    # Normalize orientations: ori_sim is array of tuples / None
    def _norm_ori(o):
        if o is None:
            return None
        arr = np.asarray(o)
        flat = arr.ravel()
        return tuple(int(v) for v in flat)

    ori_sim_norm = [_norm_ori(o) for o in ori_sim]

    sim_matches_df = pd.DataFrame(
        {
            "nv_label": nv_labels_sim.astype(int),  # if you added this earlier
            "orientation": ori_sim_norm,
            "site_index": site_index_sim.astype(int),
            "x_A": x_sim_A,
            "y_A": y_sim_A,
            "z_A": z_sim_A,
            "distance_A": np.sqrt(x_sim_A**2 + y_sim_A**2 + z_sim_A**2),
            "f0_fit_kHz": f0_sim_kHz,
            "f1_fit_kHz": f1_sim_kHz,
            "kappa": kappa_sim,
        }
    )

    # Define f_minus/f_plus in the same convention used downstream
    sim_matches_df["f_minus_kHz"] = np.minimum(
        sim_matches_df["f0_fit_kHz"], sim_matches_df["f1_fit_kHz"]
    )
    sim_matches_df["f_plus_kHz"] = np.maximum(
        sim_matches_df["f0_fit_kHz"], sim_matches_df["f1_fit_kHz"]
    )
    site_stats_sim = build_site_multiplicity_with_theory(
        matches_df=sim_matches_df,
        orbit_df=orbit_df,
        p13=0.011,  # natural abundance
    )
    multiplicity_plots(site_stats_sim, dataset_label="Simulation")
    make_a_table(site_stats_sim, topN=15, dataset_label="Simulation")

    kpl.show(block=True)
