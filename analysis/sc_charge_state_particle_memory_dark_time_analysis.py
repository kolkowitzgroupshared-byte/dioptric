import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl


from matplotlib.patches import Circle
from utils import data_manager as dm


import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl


# =============================================================================
# PARTICLE-MEMORY CHARGE-STATE ANALYSIS
#
# Main goals
# ----------
# 1. Plot NV- -> NV0 charge loss as a function of DARK WAIT TIME, not run index.
# 2. Identify anomalous runs relative to other runs at the SAME dark wait.
# 3. Measure short-range spatial correlation as a function of dark wait.
# 4. Measure distance-dependent spatial enrichment separately at each dark wait.
# 5. Plot the strongest wait-conditioned candidate events in real-space.
#
# Expected arrays
# ---------------
# counts[exp, nv, run, step, rep]
#
# rep_initial = 11 : immediate final check after initialization
# rep_final   = 12 : final readout after dark wait
#
# Coordinate convention
# ---------------------
# nv.coords[CoordsKey.PIXEL] -> [x, y] camera-pixel coordinate
# =============================================================================


# =============================================================================
# General utilities
# =============================================================================

def _robust_center_scale(values):
    """
    Return median and robust sigma = 1.4826 * MAD.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    sigma = 1.4826 * mad

    return float(med), float(sigma)


def _robust_zscore(values):
    """
    Median/MAD robust z-score.
    """
    values = np.asarray(values, dtype=float)

    med, sigma = _robust_center_scale(values)

    if not np.isfinite(sigma) or sigma <= 0:
        z = np.full(values.shape, np.nan, dtype=float)
    else:
        z = (values - med) / sigma

    return z, med, sigma


def _safe_divide(num, den):
    """
    Elementwise num/den, returning NaN where den <= 0 or non-finite.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    out = np.full(
        np.broadcast_shapes(num.shape, den.shape),
        np.nan,
        dtype=float,
    )

    good = (
        np.isfinite(num)
        & np.isfinite(den)
        & (den > 0)
    )

    out[good] = num[good] / den[good]
    return out


def _group_stats(values, group_values, unique_groups):
    """
    Group 1D values by group_values.

    Returns median, mean, std, sem, q25, q75, min, max, n.
    """
    values = np.asarray(values, dtype=float)
    group_values = np.asarray(group_values, dtype=float)
    unique_groups = np.asarray(unique_groups, dtype=float)

    stats = {
        key: []
        for key in (
            "median",
            "mean",
            "std",
            "sem",
            "q25",
            "q75",
            "min",
            "max",
            "n",
        )
    }

    for group in unique_groups:
        mask = np.isclose(group_values, group)
        vals = values[mask]
        vals = vals[np.isfinite(vals)]

        n = len(vals)
        stats["n"].append(n)

        if n == 0:
            for key in (
                "median",
                "mean",
                "std",
                "sem",
                "q25",
                "q75",
                "min",
                "max",
            ):
                stats[key].append(np.nan)
            continue

        stats["median"].append(np.nanmedian(vals))
        stats["mean"].append(np.nanmean(vals))
        stats["q25"].append(np.nanpercentile(vals, 25))
        stats["q75"].append(np.nanpercentile(vals, 75))
        stats["min"].append(np.nanmin(vals))
        stats["max"].append(np.nanmax(vals))

        if n > 1:
            std = np.nanstd(vals, ddof=1)
            sem = std / np.sqrt(n)
        else:
            std = 0.0
            sem = np.nan

        stats["std"].append(std)
        stats["sem"].append(sem)

    for key in stats:
        dtype = int if key == "n" else float
        stats[key] = np.asarray(stats[key], dtype=dtype)

    return stats


def _make_wait_jitter(
    dark_wait_by_run,
    random_seed=20260818,
):
    """
    Small visual-only x jitter for repeated runs at the same dark wait.

    The scientific x coordinate remains dark_wait_by_run.
    """
    dark_wait_by_run = np.asarray(dark_wait_by_run, dtype=float)
    unique_waits = np.sort(np.unique(dark_wait_by_run))

    rng = np.random.default_rng(random_seed)
    x_plot = dark_wait_by_run.copy()

    for wait_s in unique_waits:
        inds = np.where(np.isclose(dark_wait_by_run, wait_s))[0]

        if len(inds) <= 1:
            continue

        if wait_s == 0:
            jitter_scale = 0.75
        else:
            jitter_scale = max(0.012 * wait_s, 0.35)

        x_plot[inds] += rng.normal(
            loc=0.0,
            scale=jitter_scale,
            size=len(inds),
        )

    return x_plot


def _apply_wait_axis(ax, unique_waits):
    """
    Dark-time axis that retains t=0 while spreading 10...3600 s.
    """
    unique_waits = np.asarray(unique_waits, dtype=float)

    ax.set_xscale(
        "symlog",
        linthresh=10,
        linscale=1.0,
    )

    ax.set_xticks(unique_waits)

    ax.set_xticklabels(
        [f"{wait:g}" for wait in unique_waits],
        rotation=35,
        ha="right",
    )

    ax.set_xlabel("Dark wait time (s)")


# =============================================================================
# NV camera coordinates
# =============================================================================

def _charge_corr_get_coords(
    raw_data,
    img_coords=None,
):
    """
    Return camera pixel coordinates [nv, 2] = [x, y].

    In these datasets the image coordinates are stored as

        nv.coords[CoordsKey.PIXEL]

    We identify the PIXEL enum key by key.name so this analysis file does not
    need to import CoordsKey explicitly.
    """
    nv_list = raw_data["nv_list"]

    if img_coords is not None:
        coords_xy = np.asarray(img_coords, dtype=float)

        expected_shape = (len(nv_list), 2)

        if coords_xy.shape != expected_shape:
            raise ValueError(
                f"img_coords must have shape {expected_shape}; "
                f"got {coords_xy.shape}"
            )

        return coords_xy

    coords_xy = []

    for nv_ind, nv in enumerate(nv_list):
        coords_dict = getattr(nv, "coords", None)

        if not isinstance(coords_dict, dict):
            raise ValueError(
                f"NV {nv_ind} does not have a valid coords dictionary."
            )

        pixel_key = None

        for key in coords_dict.keys():
            if getattr(key, "name", None) == "PIXEL":
                pixel_key = key
                break

            key_string = str(key).upper()

            if (
                key_string == "PIXEL"
                or key_string.endswith(".PIXEL")
            ):
                pixel_key = key
                break

        if pixel_key is None:
            raise ValueError(
                f"Could not find PIXEL coordinates for NV {nv_ind}. "
                f"Available keys: {list(coords_dict.keys())}"
            )

        xy = np.asarray(
            coords_dict[pixel_key],
            dtype=float,
        ).ravel()

        if xy.size < 2:
            raise ValueError(
                f"Invalid PIXEL coordinate for NV {nv_ind}: {xy}"
            )

        if not np.all(np.isfinite(xy[:2])):
            raise ValueError(
                f"Non-finite PIXEL coordinate for NV {nv_ind}: {xy}"
            )

        coords_xy.append(
            [float(xy[0]), float(xy[1])]
        )

    return np.asarray(coords_xy, dtype=float)


# =============================================================================
# Pair-distance utilities
# =============================================================================

def _make_pair_bin_matrix(
    coords_xy,
    distance_bins,
    scale=1.0,
):
    """
    Precompute pair distance and distance-bin index for every NV pair.
    """
    coords_xy = np.asarray(coords_xy, dtype=float)
    coords_scaled = coords_xy * float(scale)

    dx = (
        coords_scaled[:, None, 0]
        - coords_scaled[None, :, 0]
    )

    dy = (
        coords_scaled[:, None, 1]
        - coords_scaled[None, :, 1]
    )

    distance_matrix = np.sqrt(dx**2 + dy**2)

    num_bins = len(distance_bins) - 1

    pair_bin_matrix = (
        np.digitize(
            distance_matrix,
            distance_bins,
            right=False,
        )
        - 1
    )

    invalid = (
        (pair_bin_matrix < 0)
        | (pair_bin_matrix >= num_bins)
    )

    pair_bin_matrix[invalid] = -1
    np.fill_diagonal(pair_bin_matrix, -1)

    return distance_matrix, pair_bin_matrix


def _pair_hist_from_indices(
    inds,
    pair_bin_matrix,
    num_bins,
    nv_weights=None,
):
    """
    Histogram all unique NV pairs among inds by distance bin.

    If nv_weights is supplied, pair i,j contributes weight_i * weight_j.
    """
    inds = np.asarray(inds, dtype=int)

    if len(inds) < 2:
        return np.zeros(num_bins, dtype=float)

    sub_bins = pair_bin_matrix[np.ix_(inds, inds)]

    tri_i, tri_j = np.triu_indices(
        len(inds),
        k=1,
    )

    bins = sub_bins[tri_i, tri_j]
    valid = bins >= 0
    bins = bins[valid]

    if len(bins) == 0:
        return np.zeros(num_bins, dtype=float)

    if nv_weights is None:
        return np.bincount(
            bins,
            minlength=num_bins,
        ).astype(float)

    local_weights = np.asarray(
        nv_weights,
        dtype=float,
    )[inds]

    pair_weights = (
        local_weights[tri_i]
        * local_weights[tri_j]
    )

    pair_weights = pair_weights[valid]

    return np.bincount(
        bins,
        weights=pair_weights,
        minlength=num_bins,
    ).astype(float)


# =============================================================================
# Phenomenological dark-time fit
# =============================================================================

def _loss_saturation_model(
    t,
    loss0,
    loss_inf,
    tau_s,
):
    """
    Phenomenological ensemble loss model:

        L(t) = L0 + (Linf - L0) * (1 - exp(-t/tau))

    This is useful as a compact effective timescale, but does not assume every
    NV has the same microscopic switching rate.
    """
    t = np.asarray(t, dtype=float)

    return (
        loss0
        + (loss_inf - loss0)
        * (1.0 - np.exp(-t / tau_s))
    )


def _fit_loss_vs_dark_time(
    unique_waits,
    loss_median_fraction,
):
    """
    Fit the phenomenological saturation model to median loss fraction.
    """
    x = np.asarray(unique_waits, dtype=float)
    y = np.asarray(loss_median_fraction, dtype=float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
    )

    x = x[good]
    y = y[good]

    if len(x) < 4:
        return None

    loss0_guess = float(np.clip(y[np.argmin(x)], 0, 0.95))
    loss_inf_guess = float(
        np.clip(
            max(np.nanmax(y), loss0_guess + 0.02),
            loss0_guess + 1e-3,
            0.999,
        )
    )

    positive_x = x[x > 0]

    if len(positive_x) == 0:
        return None

    tau_guess = float(np.nanmedian(positive_x))

    try:
        popt, pcov = curve_fit(
            _loss_saturation_model,
            x,
            y,
            p0=[
                loss0_guess,
                loss_inf_guess,
                tau_guess,
            ],
            bounds=(
                [0.0, 0.0, 1e-3],
                [1.0, 1.0, 1e7],
            ),
            maxfev=50000,
        )
    except Exception:
        return None

    perr = np.sqrt(np.diag(pcov))

    return {
        "loss0": float(popt[0]),
        "loss_inf": float(popt[1]),
        "tau_s": float(popt[2]),
        "loss0_err": float(perr[0]),
        "loss_inf_err": float(perr[1]),
        "tau_s_err": float(perr[2]),
        "popt": popt,
        "pcov": pcov,
    }


# =============================================================================
# Wait-conditioned anomaly statistics
# =============================================================================

def _calculate_wait_conditioned_anomalies(
    loss_fraction_by_run,
    dark_wait_by_run,
):
    """
    Compare every run only with runs at the SAME dark wait.

    This is essential when combining 0...3600 s datasets because the ordinary
    loss probability itself depends strongly on dark wait time.
    """
    loss_fraction_by_run = np.asarray(
        loss_fraction_by_run,
        dtype=float,
    )

    dark_wait_by_run = np.asarray(
        dark_wait_by_run,
        dtype=float,
    )

    unique_waits = np.sort(
        np.unique(dark_wait_by_run)
    )

    z_same_wait = np.full(
        len(loss_fraction_by_run),
        np.nan,
        dtype=float,
    )

    empirical_p_same_wait = np.full(
        len(loss_fraction_by_run),
        np.nan,
        dtype=float,
    )

    median_same_wait = np.full(
        len(loss_fraction_by_run),
        np.nan,
        dtype=float,
    )

    sigma_same_wait = np.full(
        len(loss_fraction_by_run),
        np.nan,
        dtype=float,
    )

    for wait_s in unique_waits:
        inds = np.where(
            np.isclose(
                dark_wait_by_run,
                wait_s,
            )
        )[0]

        vals = loss_fraction_by_run[inds]

        z_local, med_local, sigma_local = (
            _robust_zscore(vals)
        )

        z_same_wait[inds] = z_local
        median_same_wait[inds] = med_local
        sigma_same_wait[inds] = sigma_local

        for local_ind, global_ind in enumerate(inds):
            target = vals[local_ind]

            if not np.isfinite(target):
                continue

            others = np.delete(vals, local_ind)
            others = others[np.isfinite(others)]

            if len(others) == 0:
                continue

            empirical_p_same_wait[global_ind] = (
                1
                + np.sum(others >= target)
            ) / (
                len(others) + 1
            )

    return {
        "z": z_same_wait,
        "empirical_p": empirical_p_same_wait,
        "median": median_same_wait,
        "sigma": sigma_same_wait,
    }


# =============================================================================
# Spatial correlation, evaluated separately for each dark wait
# =============================================================================

def _analyze_spatial_correlations_by_wait(
    eligible_mask,
    switch_mask,
    dark_wait_by_run,
    pair_bin_matrix,
    distance_bins,
    short_range_max,
    alpha_prior=0.5,
    beta_prior=0.5,
    n_permutations=2000,
    random_seed=12345,
    verbose=True,
):
    """
    Calculate spatial statistics separately at each dark wait.

    Null 1: independent-NV switching
        Each eligible NV switches with its own p_i estimated from runs at the
        SAME dark wait.

    Null 2: conditional spatial null
        Preserve the observed K_r exactly for each run and randomize which
        eligible NVs switched, weighted by p_i.

    The returned run-level short-range z-score asks:
        Given K_r, is this particular run more spatially clustered than
        randomized runs with the same event size?
    """
    eligible_mask = np.asarray(
        eligible_mask,
        dtype=bool,
    )

    switch_mask = np.asarray(
        switch_mask,
        dtype=bool,
    )

    dark_wait_by_run = np.asarray(
        dark_wait_by_run,
        dtype=float,
    )

    unique_waits = np.sort(
        np.unique(dark_wait_by_run)
    )

    num_nvs, total_runs = switch_mask.shape
    num_bins = len(distance_bins) - 1

    distance_centers = 0.5 * (
        distance_bins[:-1]
        + distance_bins[1:]
    )

    short_bins = (
        distance_bins[1:]
        <= (
            short_range_max
            + 1e-12
        )
    )

    if not np.any(short_bins):
        raise ValueError(
            f"No distance bins fall fully inside 0-{short_range_max:g}."
        )

    num_waits = len(unique_waits)

    observed_pair_counts_by_wait = np.zeros(
        (num_waits, num_bins),
        dtype=float,
    )

    independent_expected_pairs_by_wait = np.zeros(
        (num_waits, num_bins),
        dtype=float,
    )

    conditional_mean_pairs_by_wait = np.full(
        (num_waits, num_bins),
        np.nan,
        dtype=float,
    )

    independent_p_by_wait_bin = np.full(
        (num_waits, num_bins),
        np.nan,
        dtype=float,
    )

    conditional_p_by_wait_bin = np.full(
        (num_waits, num_bins),
        np.nan,
        dtype=float,
    )

    spatial_enrichment_by_wait = np.full(
        (num_waits, num_bins),
        np.nan,
        dtype=float,
    )

    coincidence_enhancement_by_wait = np.full(
        (num_waits, num_bins),
        np.nan,
        dtype=float,
    )

    short_observed_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    short_independent_mean_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    short_conditional_mean_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    short_g_independent_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    short_spatial_enrichment_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    short_p_independent_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    short_p_spatial_by_wait = np.full(
        num_waits,
        np.nan,
        dtype=float,
    )

    nv_switch_probability_by_wait = np.full(
        (num_nvs, num_waits),
        np.nan,
        dtype=float,
    )

    # Per-run spatial metrics
    short_pairs_observed_by_run = np.full(
        total_runs,
        np.nan,
        dtype=float,
    )

    short_pairs_cond_mean_by_run = np.full(
        total_runs,
        np.nan,
        dtype=float,
    )

    short_pairs_cond_std_by_run = np.full(
        total_runs,
        np.nan,
        dtype=float,
    )

    short_spatial_z_by_run = np.full(
        total_runs,
        np.nan,
        dtype=float,
    )

    short_spatial_p_by_run = np.full(
        total_runs,
        np.nan,
        dtype=float,
    )

    rng = np.random.default_rng(
        random_seed
    )

    for wait_ind, wait_s in enumerate(
        unique_waits
    ):
        run_inds = np.where(
            np.isclose(
                dark_wait_by_run,
                wait_s,
            )
        )[0]

        eligible_w = eligible_mask[
            :,
            run_inds,
        ]

        switch_w = switch_mask[
            :,
            run_inds,
        ]

        # --------------------------------------------------------------
        # Per-NV switching probability at THIS dark wait
        # --------------------------------------------------------------

        nv_eligible_trials = np.sum(
            eligible_w,
            axis=1,
        ).astype(float)

        nv_switch_trials = np.sum(
            switch_w,
            axis=1,
        ).astype(float)

        p_nv = (
            nv_switch_trials
            + alpha_prior
        ) / (
            nv_eligible_trials
            + alpha_prior
            + beta_prior
        )

        nv_switch_probability_by_wait[
            :,
            wait_ind,
        ] = p_nv

        # --------------------------------------------------------------
        # Observed + independent expectation
        # --------------------------------------------------------------

        observed_hist = np.zeros(
            num_bins,
            dtype=float,
        )

        independent_expected_hist = np.zeros(
            num_bins,
            dtype=float,
        )

        eligible_inds_by_run = []
        switched_inds_by_run = []

        for global_run_ind in run_inds:
            eligible_inds = np.where(
                eligible_mask[
                    :,
                    global_run_ind,
                ]
            )[0]

            switched_inds = np.where(
                switch_mask[
                    :,
                    global_run_ind,
                ]
            )[0]

            eligible_inds_by_run.append(
                eligible_inds
            )

            switched_inds_by_run.append(
                switched_inds
            )

            obs_hist_run = _pair_hist_from_indices(
                switched_inds,
                pair_bin_matrix,
                num_bins,
            )

            observed_hist += obs_hist_run

            independent_expected_hist += (
                _pair_hist_from_indices(
                    eligible_inds,
                    pair_bin_matrix,
                    num_bins,
                    nv_weights=p_nv,
                )
            )

            short_pairs_observed_by_run[
                global_run_ind
            ] = np.sum(
                obs_hist_run[
                    short_bins
                ]
            )

        observed_pair_counts_by_wait[
            wait_ind
        ] = observed_hist

        independent_expected_pairs_by_wait[
            wait_ind
        ] = independent_expected_hist

        # --------------------------------------------------------------
        # Monte Carlo
        # --------------------------------------------------------------

        null_independent = np.zeros(
            (
                n_permutations,
                num_bins,
            ),
            dtype=float,
        )

        null_conditional = np.zeros(
            (
                n_permutations,
                num_bins,
            ),
            dtype=float,
        )

        # Store conditional short-range result for every run.
        cond_short_by_run = np.zeros(
            (
                n_permutations,
                len(run_inds),
            ),
            dtype=float,
        )

        if verbose:
            print(
                f"  spatial null: wait={wait_s:g} s | "
                f"runs={len(run_inds)} | "
                f"permutations={n_permutations}"
            )

        for perm_ind in range(
            n_permutations
        ):
            hist_independent = np.zeros(
                num_bins,
                dtype=float,
            )

            hist_conditional = np.zeros(
                num_bins,
                dtype=float,
            )

            for local_run_ind, global_run_ind in enumerate(
                run_inds
            ):
                eligible_inds = (
                    eligible_inds_by_run[
                        local_run_ind
                    ]
                )

                n_eligible = len(
                    eligible_inds
                )

                if n_eligible < 2:
                    continue

                probs = p_nv[
                    eligible_inds
                ]

                # ------------------------------------------------------
                # Null 1: independent NV switching
                # ------------------------------------------------------

                draws = (
                    rng.random(
                        n_eligible
                    )
                    < probs
                )

                simulated_switch_inds = (
                    eligible_inds[
                        draws
                    ]
                )

                hist_independent += (
                    _pair_hist_from_indices(
                        simulated_switch_inds,
                        pair_bin_matrix,
                        num_bins,
                    )
                )

                # ------------------------------------------------------
                # Null 2: preserve observed K exactly
                # ------------------------------------------------------

                observed_k = int(
                    np.sum(
                        switch_mask[
                            :,
                            global_run_ind,
                        ]
                    )
                )

                if observed_k < 2:
                    continue

                observed_k = min(
                    observed_k,
                    n_eligible,
                )

                choice_weights = np.clip(
                    probs.copy(),
                    1e-12,
                    None,
                )

                choice_weights /= np.sum(
                    choice_weights
                )

                randomized_inds = rng.choice(
                    eligible_inds,
                    size=observed_k,
                    replace=False,
                    p=choice_weights,
                )

                hist_this_run = (
                    _pair_hist_from_indices(
                        randomized_inds,
                        pair_bin_matrix,
                        num_bins,
                    )
                )

                hist_conditional += hist_this_run

                cond_short_by_run[
                    perm_ind,
                    local_run_ind,
                ] = np.sum(
                    hist_this_run[
                        short_bins
                    ]
                )

            null_independent[
                perm_ind
            ] = hist_independent

            null_conditional[
                perm_ind
            ] = hist_conditional

        # --------------------------------------------------------------
        # Distance-dependent statistics
        # --------------------------------------------------------------

        cond_mean = np.mean(
            null_conditional,
            axis=0,
        )

        conditional_mean_pairs_by_wait[
            wait_ind
        ] = cond_mean

        independent_p_by_wait_bin[
            wait_ind
        ] = (
            1
            + np.sum(
                null_independent
                >= observed_hist[
                    None,
                    :
                ],
                axis=0,
            )
        ) / (
            n_permutations
            + 1
        )

        conditional_p_by_wait_bin[
            wait_ind
        ] = (
            1
            + np.sum(
                null_conditional
                >= observed_hist[
                    None,
                    :
                ],
                axis=0,
            )
        ) / (
            n_permutations
            + 1
        )

        good_ind = (
            independent_expected_hist
            > 0
        )

        coincidence_enhancement_by_wait[
            wait_ind,
            good_ind,
        ] = (
            observed_hist[
                good_ind
            ]
            / independent_expected_hist[
                good_ind
            ]
        )

        good_cond = cond_mean > 0

        spatial_enrichment_by_wait[
            wait_ind,
            good_cond,
        ] = (
            observed_hist[
                good_cond
            ]
            / cond_mean[
                good_cond
            ]
        )

        # --------------------------------------------------------------
        # Short-range aggregate
        # --------------------------------------------------------------

        observed_short = float(
            np.sum(
                observed_hist[
                    short_bins
                ]
            )
        )

        ind_short_null = np.sum(
            null_independent[
                :,
                short_bins,
            ],
            axis=1,
        )

        cond_short_null = np.sum(
            null_conditional[
                :,
                short_bins,
            ],
            axis=1,
        )

        ind_short_mean = float(
            np.mean(
                ind_short_null
            )
        )

        cond_short_mean = float(
            np.mean(
                cond_short_null
            )
        )

        short_observed_by_wait[
            wait_ind
        ] = observed_short

        short_independent_mean_by_wait[
            wait_ind
        ] = ind_short_mean

        short_conditional_mean_by_wait[
            wait_ind
        ] = cond_short_mean

        if ind_short_mean > 0:
            short_g_independent_by_wait[
                wait_ind
            ] = (
                observed_short
                / ind_short_mean
            )

        if cond_short_mean > 0:
            short_spatial_enrichment_by_wait[
                wait_ind
            ] = (
                observed_short
                / cond_short_mean
            )

        short_p_independent_by_wait[
            wait_ind
        ] = (
            1
            + np.sum(
                ind_short_null
                >= observed_short
            )
        ) / (
            n_permutations
            + 1
        )

        short_p_spatial_by_wait[
            wait_ind
        ] = (
            1
            + np.sum(
                cond_short_null
                >= observed_short
            )
        ) / (
            n_permutations
            + 1
        )

        # --------------------------------------------------------------
        # Per-run spatial clustering significance
        # --------------------------------------------------------------

        for local_run_ind, global_run_ind in enumerate(
            run_inds
        ):
            null_vals = cond_short_by_run[
                :,
                local_run_ind,
            ]

            observed_val = (
                short_pairs_observed_by_run[
                    global_run_ind
                ]
            )

            mean_null = np.mean(
                null_vals
            )

            std_null = np.std(
                null_vals,
                ddof=1,
            )

            short_pairs_cond_mean_by_run[
                global_run_ind
            ] = mean_null

            short_pairs_cond_std_by_run[
                global_run_ind
            ] = std_null

            if (
                np.isfinite(std_null)
                and std_null > 0
            ):
                short_spatial_z_by_run[
                    global_run_ind
                ] = (
                    observed_val
                    - mean_null
                ) / std_null

            short_spatial_p_by_run[
                global_run_ind
            ] = (
                1
                + np.sum(
                    null_vals
                    >= observed_val
                )
            ) / (
                n_permutations
                + 1
            )

    return {
        "unique_waits": unique_waits,
        "distance_centers": distance_centers,
        "short_bins": short_bins,

        "observed_pair_counts_by_wait":
            observed_pair_counts_by_wait,

        "independent_expected_pairs_by_wait":
            independent_expected_pairs_by_wait,

        "conditional_mean_pairs_by_wait":
            conditional_mean_pairs_by_wait,

        "coincidence_enhancement_by_wait":
            coincidence_enhancement_by_wait,

        "spatial_enrichment_by_wait":
            spatial_enrichment_by_wait,

        "independent_p_by_wait_bin":
            independent_p_by_wait_bin,

        "conditional_p_by_wait_bin":
            conditional_p_by_wait_bin,

        "short_observed_by_wait":
            short_observed_by_wait,

        "short_independent_mean_by_wait":
            short_independent_mean_by_wait,

        "short_conditional_mean_by_wait":
            short_conditional_mean_by_wait,

        "short_g_independent_by_wait":
            short_g_independent_by_wait,

        "short_spatial_enrichment_by_wait":
            short_spatial_enrichment_by_wait,

        "short_p_independent_by_wait":
            short_p_independent_by_wait,

        "short_p_spatial_by_wait":
            short_p_spatial_by_wait,

        "nv_switch_probability_by_wait":
            nv_switch_probability_by_wait,

        "short_pairs_observed_by_run":
            short_pairs_observed_by_run,

        "short_pairs_cond_mean_by_run":
            short_pairs_cond_mean_by_run,

        "short_pairs_cond_std_by_run":
            short_pairs_cond_std_by_run,

        "short_spatial_z_by_run":
            short_spatial_z_by_run,

        "short_spatial_p_by_run":
            short_spatial_p_by_run,
    }


# =============================================================================
# Main analysis
# =============================================================================

def analyze_charge_memory_vs_dark_time(
    file_stems,
    selected_waits_s=None,
    rep_initial=11,
    rep_final=12,

    exclude_nv_inds=None,

    # Charge classification
    initial_margin_counts=0.0,
    final_margin_counts=0.0,

    # Coordinates / physical distance
    img_coords=None,
    um_per_pixel=0.43,

    # Spatial-correlation distance bins
    distance_bin_width=10.0,
    distance_max=None,
    short_range_max=30.0,

    # Monte-Carlo null
    alpha_prior=0.5,
    beta_prior=0.5,
    n_permutations=2000,

    # Plot controls
    top_n_event_maps=6,
    annotate_top_n=8,

    random_seed=12345,
    verbose=True,
):
    """
    Full dark-time-dependent charge-memory analysis.

    IMPORTANT
    ---------
    Anomalous runs are compared to other runs at the SAME dark wait.

    Spatial null models are also built separately at each dark wait.
    This avoids conflating the expected increase of charge loss with time
    with a rare correlated event.
    """
    file_stems = list(file_stems)

    if selected_waits_s is not None:
        selected_waits_s = np.asarray(
            selected_waits_s,
            dtype=float,
        )

    if exclude_nv_inds is None:
        exclude_nv_inds = np.array(
            [],
            dtype=int,
        )
    else:
        exclude_nv_inds = np.unique(
            np.asarray(
                exclude_nv_inds,
                dtype=int,
            )
        )

    datasets = []

    switch_blocks = []
    eligible_blocks = []

    global_file_index = []
    global_local_run = []

    master_num_nvs = None
    master_keep = None
    master_coords = None
    master_original_nv_inds = None

    global_offset = 0

    # =====================================================================
    # Load files
    # =====================================================================

    for file_ind, file_stem in enumerate(
        file_stems
    ):
        raw_data = dm.get_raw_data(
            file_stem=file_stem,
            load_npz=True,
        )

        wait_s = float(
            raw_data["dark_wait_s"]
        )

        if selected_waits_s is not None:
            if not np.any(
                np.isclose(
                    wait_s,
                    selected_waits_s,
                )
            ):
                continue

        counts_all = np.asarray(
            raw_data["counts"],
            dtype=float,
        )

        if counts_all.ndim != 5:
            raise ValueError(
                "Expected counts[exp,nv,run,step,rep], "
                f"got {counts_all.shape}"
            )

        counts = counts_all[
            0,
            :,
            :,
            0,
            :,
        ]

        num_nvs, num_runs, num_reps = (
            counts.shape
        )

        if master_num_nvs is None:
            master_num_nvs = num_nvs
        elif num_nvs != master_num_nvs:
            raise ValueError(
                "All files must contain the same NV list length. "
                f"Expected {master_num_nvs}, got {num_nvs} in {file_stem}"
            )

        if not (
            0 <= rep_initial < num_reps
            and 0 <= rep_final < num_reps
        ):
            raise ValueError(
                f"Invalid rep indices for {file_stem}. "
                f"num_reps={num_reps}"
            )

        if "analysis_thresholds" in raw_data:
            thresholds = np.asarray(
                raw_data["analysis_thresholds"],
                dtype=float,
            )
        elif "thresholds" in raw_data:
            thresholds = np.asarray(
                raw_data["thresholds"],
                dtype=float,
            )
        else:
            raise ValueError(
                f"No saved thresholds in {file_stem}"
            )

        if thresholds.shape != (num_nvs,):
            raise ValueError(
                f"Threshold shape mismatch in {file_stem}: "
                f"{thresholds.shape}"
            )

        # --------------------------------------------------------------
        # Consistent NV keep mask
        # --------------------------------------------------------------

        if master_keep is None:
            master_keep = np.ones(
                num_nvs,
                dtype=bool,
            )

            valid_excluded = exclude_nv_inds[
                (exclude_nv_inds >= 0)
                & (exclude_nv_inds < num_nvs)
            ]

            master_keep[
                valid_excluded
            ] = False

            master_original_nv_inds = (
                np.arange(
                    num_nvs
                )[master_keep]
            )

        counts = counts[
            master_keep,
            :,
            :,
        ]

        thresholds = thresholds[
            master_keep
        ]

        # --------------------------------------------------------------
        # Pixel coordinates
        # --------------------------------------------------------------

        coords_this = _charge_corr_get_coords(
            raw_data,
            img_coords=img_coords,
        )

        coords_this = coords_this[
            master_keep
        ]

        if master_coords is None:
            master_coords = coords_this.copy()

        else:
            # Translation is okay; deformation / reordering is not.
            delta = (
                coords_this
                - master_coords
            )

            median_delta = np.nanmedian(
                delta,
                axis=0,
            )

            residual = (
                delta
                - median_delta[
                    None,
                    :
                ]
            )

            residual_mag = np.sqrt(
                np.sum(
                    residual**2,
                    axis=1,
                )
            )

            median_residual = float(
                np.nanmedian(
                    residual_mag
                )
            )

            if (
                verbose
                and median_residual > 2.0
            ):
                print(
                    "WARNING: relative NV coordinates changed by "
                    f"median {median_residual:.2f} camera px in {file_stem}. "
                    "Check NV ordering."
                )

        # --------------------------------------------------------------
        # Charge states
        # --------------------------------------------------------------

        c11 = counts[
            :,
            :,
            rep_initial,
        ]

        c12 = counts[
            :,
            :,
            rep_final,
        ]

        eligible = (
            c11
            > (
                thresholds[:, None]
                + initial_margin_counts
            )
        )

        final_nv0 = (
            c12
            <= (
                thresholds[:, None]
                - final_margin_counts
            )
        )

        switch = (
            eligible
            & final_nv0
        )

        switch_blocks.append(
            switch
        )

        eligible_blocks.append(
            eligible
        )

        global_runs = (
            global_offset
            + np.arange(
                num_runs,
                dtype=int,
            )
        )

        datasets.append(
            {
                "file_ind":
                    file_ind,

                "file_stem":
                    file_stem,

                "dark_wait_s":
                    wait_s,

                "num_runs":
                    num_runs,

                "global_runs":
                    global_runs,

                "global_start":
                    int(global_offset),

                "global_stop":
                    int(
                        global_offset
                        + num_runs
                    ),
            }
        )

        global_file_index.extend(
            [file_ind] * num_runs
        )

        global_local_run.extend(
            range(num_runs)
        )

        global_offset += num_runs

    if len(datasets) == 0:
        raise ValueError(
            "No matching datasets loaded."
        )

    # =====================================================================
    # Concatenate run axis
    # =====================================================================

    switch_mask = np.concatenate(
        switch_blocks,
        axis=1,
    )

    eligible_mask = np.concatenate(
        eligible_blocks,
        axis=1,
    )

    num_kept_nvs, total_runs = (
        switch_mask.shape
    )

    global_runs = np.arange(
        total_runs,
        dtype=int,
    )

    global_file_index = np.asarray(
        global_file_index,
        dtype=int,
    )

    global_local_run = np.asarray(
        global_local_run,
        dtype=int,
    )

    dark_wait_by_run = np.concatenate(
        [
            np.full(
                dataset["num_runs"],
                dataset["dark_wait_s"],
                dtype=float,
            )
            for dataset in datasets
        ]
    )

    unique_waits = np.sort(
        np.unique(
            dark_wait_by_run
        )
    )

    # =====================================================================
    # Per-run charge-loss statistics
    # =====================================================================

    eligible_by_run = np.sum(
        eligible_mask,
        axis=0,
    ).astype(int)

    switches_by_run = np.sum(
        switch_mask,
        axis=0,
    ).astype(int)

    loss_fraction_by_run = _safe_divide(
        switches_by_run,
        eligible_by_run,
    )

    retention_by_run = (
        1.0
        - loss_fraction_by_run
    )

    # --------------------------------------------------------------
    # Wait-conditioned anomaly score
    # --------------------------------------------------------------

    wait_anomaly = (
        _calculate_wait_conditioned_anomalies(
            loss_fraction_by_run,
            dark_wait_by_run,
        )
    )

    loss_z_same_wait = (
        wait_anomaly["z"]
    )

    loss_empirical_p_same_wait = (
        wait_anomaly[
            "empirical_p"
        ]
    )

    # =====================================================================
    # Grouped dark-time statistics
    # =====================================================================

    loss_stats = _group_stats(
        100.0
        * loss_fraction_by_run,
        dark_wait_by_run,
        unique_waits,
    )

    loss_fraction_stats = _group_stats(
        loss_fraction_by_run,
        dark_wait_by_run,
        unique_waits,
    )

    retention_stats = _group_stats(
        100.0
        * retention_by_run,
        dark_wait_by_run,
        unique_waits,
    )

    switch_stats = _group_stats(
        switches_by_run,
        dark_wait_by_run,
        unique_waits,
    )

    eligible_stats = _group_stats(
        eligible_by_run,
        dark_wait_by_run,
        unique_waits,
    )

    # --------------------------------------------------------------
    # Phenomenological dark-time fit
    # --------------------------------------------------------------

    loss_fit = _fit_loss_vs_dark_time(
        unique_waits,
        loss_fraction_stats[
            "median"
        ],
    )

    # =====================================================================
    # Pair-distance geometry
    # =====================================================================

    if um_per_pixel is None:
        distance_scale = 1.0
        distance_unit = "camera px"
    else:
        distance_scale = float(
            um_per_pixel
        )
        distance_unit = "µm"

    coords_scaled = (
        master_coords
        * distance_scale
    )

    dx_all = (
        coords_scaled[:, None, 0]
        - coords_scaled[None, :, 0]
    )

    dy_all = (
        coords_scaled[:, None, 1]
        - coords_scaled[None, :, 1]
    )

    max_possible_distance = float(
        np.nanmax(
            np.sqrt(
                dx_all**2
                + dy_all**2
            )
        )
    )

    if distance_max is None:
        distance_max_use = (
            max_possible_distance
        )
    else:
        distance_max_use = min(
            float(distance_max),
            max_possible_distance,
        )

    distance_bins = np.arange(
        0.0,
        distance_max_use
        + distance_bin_width,
        distance_bin_width,
        dtype=float,
    )

    if len(distance_bins) < 2:
        raise ValueError(
            "Invalid distance-bin configuration."
        )

    if (
        distance_bins[-1]
        < distance_max_use
    ):
        distance_bins = np.append(
            distance_bins,
            distance_max_use,
        )

    distance_centers = 0.5 * (
        distance_bins[:-1]
        + distance_bins[1:]
    )

    (
        distance_matrix,
        pair_bin_matrix,
    ) = _make_pair_bin_matrix(
        master_coords,
        distance_bins,
        scale=distance_scale,
    )

    # =====================================================================
    # Spatial correlation separately at each wait
    # =====================================================================

    if verbose:
        print(
            "\nRunning WAIT-CONDITIONED spatial null simulations..."
        )
        print(
            f"  total runs = {total_runs}"
        )
        print(
            f"  NVs = {num_kept_nvs}"
        )
        print(
            f"  waits = {unique_waits.tolist()}"
        )

    spatial = _analyze_spatial_correlations_by_wait(
        eligible_mask=eligible_mask,
        switch_mask=switch_mask,
        dark_wait_by_run=dark_wait_by_run,
        pair_bin_matrix=pair_bin_matrix,
        distance_bins=distance_bins,
        short_range_max=short_range_max,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        n_permutations=n_permutations,
        random_seed=random_seed,
        verbose=verbose,
    )

    # =====================================================================
    # Rank candidate runs by WAIT-CONDITIONED anomaly score
    # =====================================================================

    finite_z = np.where(
        np.isfinite(
            loss_z_same_wait
        )
    )[0]

    if len(finite_z) > 0:
        ranked_by_loss_z = finite_z[
            np.argsort(
                loss_z_same_wait[
                    finite_z
                ]
            )[::-1]
        ]
    else:
        ranked_by_loss_z = np.argsort(
            loss_fraction_by_run
        )[::-1]

    top_annotate = ranked_by_loss_z[
        :min(
            annotate_top_n,
            len(ranked_by_loss_z),
        )
    ]

    top_event_runs = ranked_by_loss_z[
        :min(
            top_n_event_maps,
            len(ranked_by_loss_z),
        )
    ]

    # =====================================================================
    # Plot preparation
    # =====================================================================

    figures = {}

    x_plot = _make_wait_jitter(
        dark_wait_by_run,
        random_seed=20260818,
    )

    # =====================================================================
    # FIGURE 1: Main publication-style loss vs dark time
    # =====================================================================

    fig, ax = plt.subplots(
        figsize=(8.8, 5.8)
    )

    ax.scatter(
        x_plot,
        100.0
        * loss_fraction_by_run,
        s=24,
        alpha=0.35,
        label="individual runs",
    )

    lower_err = (
        loss_stats["median"]
        - loss_stats["q25"]
    )

    upper_err = (
        loss_stats["q75"]
        - loss_stats["median"]
    )

    ax.errorbar(
        unique_waits,
        loss_stats["median"],
        yerr=[
            lower_err,
            upper_err,
        ],
        fmt="o-",
        markersize=6,
        linewidth=1.8,
        capsize=3,
        label="median ± IQR",
    )

    if loss_fit is not None:
        x_fit = np.concatenate(
            (
                np.array([0.0]),
                np.geomspace(
                    max(
                        1e-2,
                        np.min(
                            unique_waits[
                                unique_waits > 0
                            ]
                        )
                    ),
                    np.max(
                        unique_waits
                    ),
                    300,
                ),
            )
        )

        y_fit = 100.0 * (
            _loss_saturation_model(
                x_fit,
                *loss_fit["popt"],
            )
        )

        ax.plot(
            x_fit,
            y_fit,
            "--",
            linewidth=1.5,
            label=(
                "phenomenological fit: "
                f"τ={loss_fit['tau_s']:.0f} s"
            ),
        )

    for ind in top_annotate:
        if (
            np.isfinite(
                loss_z_same_wait[
                    ind
                ]
            )
            and loss_z_same_wait[ind] >= 3.0
        ):
            ax.annotate(
                (
                    f"z={loss_z_same_wait[ind]:.1f}\n"
                    f"R{global_local_run[ind]}"
                ),
                (
                    x_plot[ind],
                    100.0
                    * loss_fraction_by_run[
                        ind
                    ],
                ),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7,
            )

    _apply_wait_axis(
        ax,
        unique_waits,
    )

    ax.set_ylabel(
        "NV$^-$ → NV$^0$ loss probability (%)"
    )

    ax.set_title(
        "NV charge-state loss vs dark wait time"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    figures[
        "loss_vs_dark_time"
    ] = fig

    # =====================================================================
    # FIGURE 2: Dark-time summary
    # =====================================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.5, 10),
        sharex=True,
    )

    # A. Retention
    ax = axes[0]

    ax.scatter(
        x_plot,
        100.0
        * retention_by_run,
        s=22,
        alpha=0.30,
        label="individual runs",
    )

    ax.plot(
        unique_waits,
        retention_stats[
            "median"
        ],
        "o-",
        linewidth=1.8,
        markersize=6,
        label="median",
    )

    ax.fill_between(
        unique_waits,
        retention_stats["q25"],
        retention_stats["q75"],
        alpha=0.18,
        label="25–75%",
    )

    ax.set_ylabel(
        "NV$^-$ retention (%)"
    )

    ax.set_title(
        "True charge retention conditioned on NV$^-$ at rep 11"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    # B. Number switched
    ax = axes[1]

    ax.scatter(
        x_plot,
        switches_by_run,
        s=22,
        alpha=0.30,
    )

    ax.plot(
        unique_waits,
        switch_stats["median"],
        "o-",
        linewidth=1.8,
        markersize=6,
        label="median",
    )

    ax.fill_between(
        unique_waits,
        switch_stats["q25"],
        switch_stats["q75"],
        alpha=0.18,
    )

    ax.set_ylabel(
        "NV$^-$ → NV$^0$\ncount"
    )

    ax.set_title(
        "Number of charge-loss transitions per exposure"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    # C. Initial eligible population
    ax = axes[2]

    ax.scatter(
        x_plot,
        eligible_by_run,
        s=22,
        alpha=0.30,
    )

    ax.plot(
        unique_waits,
        eligible_stats["median"],
        "o-",
        linewidth=1.8,
        markersize=6,
        label="median",
    )

    ax.fill_between(
        unique_waits,
        eligible_stats["q25"],
        eligible_stats["q75"],
        alpha=0.18,
    )

    ax.set_ylabel(
        "Initial NV$^-$ count"
    )

    ax.set_title(
        "Initialization stability before the dark wait"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    _apply_wait_axis(
        axes[-1],
        unique_waits,
    )

    figures[
        "dark_time_summary"
    ] = fig

    # =====================================================================
    # FIGURE 3: Wait-conditioned anomalous-event score
    # =====================================================================

    fig, ax = plt.subplots(
        figsize=(8.8, 5.5)
    )

    ax.scatter(
        x_plot,
        loss_z_same_wait,
        s=30,
        alpha=0.55,
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        3.0,
        linestyle="--",
        linewidth=1.0,
        label="z = 3",
    )

    ax.axhline(
        5.0,
        linestyle="--",
        linewidth=1.0,
        label="z = 5",
    )

    for ind in top_annotate:
        if np.isfinite(
            loss_z_same_wait[
                ind
            ]
        ):
            ax.annotate(
                (
                    f"{dark_wait_by_run[ind]:g}s "
                    f"R{global_local_run[ind]}"
                ),
                (
                    x_plot[ind],
                    loss_z_same_wait[ind],
                ),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7,
            )

    _apply_wait_axis(
        ax,
        unique_waits,
    )

    ax.set_ylabel(
        "Loss anomaly z-score\n(relative to same dark wait)"
    )

    ax.set_title(
        "Rare-event candidates after removing ordinary dark-time dependence"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    figures[
        "wait_conditioned_anomalies"
    ] = fig

    # =====================================================================
    # FIGURE 4: Short-range correlation vs dark time
    # =====================================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.5, 9.5),
        sharex=True,
    )

    # A. Conditional spatial enrichment
    ax = axes[0]

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.plot(
        unique_waits,
        spatial[
            "short_spatial_enrichment_by_wait"
        ],
        "o-",
        linewidth=1.8,
        markersize=6,
    )

    ax.set_ylabel(
        "Observed / conditional null"
    )

    ax.set_title(
        f"Short-range spatial clustering (0–{short_range_max:g} {distance_unit})"
    )

    ax.grid(alpha=0.2)

    # B. Spatial Monte-Carlo p
    ax = axes[1]

    ax.axhline(
        0.05,
        linestyle="--",
        linewidth=1.0,
        label="p = 0.05",
    )

    ax.plot(
        unique_waits,
        spatial[
            "short_p_spatial_by_wait"
        ],
        "o-",
        linewidth=1.8,
        markersize=6,
    )

    ax.set_ylabel(
        "Conditional spatial p"
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_title(
        "Probability of equal-or-stronger clustering under same-K null"
    )

    ax.grid(alpha=0.2)
    ax.legend()

    # C. Independent coincidence enhancement
    ax = axes[2]

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.plot(
        unique_waits,
        spatial[
            "short_g_independent_by_wait"
        ],
        "o-",
        linewidth=1.8,
        markersize=6,
    )

    ax.set_ylabel(
        "Observed / independent null"
    )

    ax.set_title(
        "Excess simultaneous switching beyond independent-NV background"
    )

    ax.grid(alpha=0.2)

    _apply_wait_axis(
        axes[-1],
        unique_waits,
    )

    figures[
        "short_range_vs_dark_time"
    ] = fig

    # =====================================================================
    # FIGURE 5: Distance × dark-time spatial-enrichment heat map
    # =====================================================================

    enrichment_map = spatial[
        "spatial_enrichment_by_wait"
    ]

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2)
    )

    im = ax.imshow(
        enrichment_map,
        aspect="auto",
        origin="lower",
        extent=[
            distance_bins[0],
            distance_bins[-1],
            -0.5,
            len(unique_waits) - 0.5,
        ],
        interpolation="nearest",
        vmin=0.9,
        vmax=1.1,
    )

    ax.set_yticks(
        np.arange(
            len(unique_waits)
        )
    )

    ax.set_yticklabels(
        [
            f"{wait:g}"
            for wait in unique_waits
        ]
    )

    ax.set_xlabel(
        f"NV–NV separation ({distance_unit})"
    )

    ax.set_ylabel(
        "Dark wait time (s)"
    )

    ax.set_title(
        "Spatial enrichment vs separation and dark wait\n"
        "(observed / same-K conditional null)"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Spatial enrichment"
    )

    figures[
        "spatial_enrichment_heatmap"
    ] = fig

    # =====================================================================
    # FIGURE 6: Per-NV switching-probability heat map
    # =====================================================================

    nv_prob = spatial[
        "nv_switch_probability_by_wait"
    ]

    # Sort by long-time / average loss propensity.
    sort_score = np.nanmean(
        nv_prob,
        axis=1,
    )

    nv_order = np.argsort(
        sort_score
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 8.0)
    )

    im = ax.imshow(
        nv_prob[
            nv_order,
            :
        ],
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.0,
        vmax=min(
            1.0,
            float(
                np.nanpercentile(
                    nv_prob,
                    99,
                )
            )
            if np.any(
                np.isfinite(nv_prob)
            )
            else 1.0,
        ),
    )

    ax.set_xticks(
        np.arange(
            len(unique_waits)
        )
    )

    ax.set_xticklabels(
        [
            f"{wait:g}"
            for wait in unique_waits
        ],
        rotation=35,
        ha="right",
    )

    ax.set_xlabel(
        "Dark wait time (s)"
    )

    ax.set_ylabel(
        "NV rank (sorted by mean switching propensity)"
    )

    ax.set_title(
        "Per-NV NV$^-$→NV$^0$ probability vs dark wait"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Estimated switching probability"
    )

    figures[
        "nv_switch_probability_heatmap"
    ] = fig

    # =====================================================================
    # FIGURE 7: Per-run spatial clustering vs charge-loss anomaly
    # =====================================================================

    fig, ax = plt.subplots(
        figsize=(7.5, 6.0)
    )

    good = (
        np.isfinite(
            loss_z_same_wait
        )
        & np.isfinite(
            spatial[
                "short_spatial_z_by_run"
            ]
        )
    )

    ax.scatter(
        loss_z_same_wait[
            good
        ],
        spatial[
            "short_spatial_z_by_run"
        ][
            good
        ],
        s=30,
        alpha=0.55,
    )

    ax.axvline(
        3.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.axhline(
        3.0,
        linestyle="--",
        linewidth=1.0,
    )

    for ind in top_annotate:
        if (
            np.isfinite(
                loss_z_same_wait[
                    ind
                ]
            )
            and np.isfinite(
                spatial[
                    "short_spatial_z_by_run"
                ][
                    ind
                ]
            )
        ):
            ax.annotate(
                (
                    f"{dark_wait_by_run[ind]:g}s "
                    f"R{global_local_run[ind]}"
                ),
                (
                    loss_z_same_wait[ind],
                    spatial[
                        "short_spatial_z_by_run"
                    ][
                        ind
                    ],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_xlabel(
        "Charge-loss anomaly z\n(relative to same dark wait)"
    )

    ax.set_ylabel(
        f"0–{short_range_max:g} {distance_unit} spatial-clustering z"
    )

    ax.set_title(
        "Are the high-loss runs also spatially correlated?"
    )

    ax.grid(alpha=0.2)

    figures[
        "loss_anomaly_vs_spatial_clustering"
    ] = fig

    # =====================================================================
    # FIGURE 8: Event maps for strongest WAIT-CONDITIONED events
    # =====================================================================

    if len(top_event_runs) > 0:
        ncols = min(
            3,
            len(top_event_runs),
        )

        nrows = int(
            np.ceil(
                len(top_event_runs)
                / ncols
            )
        )

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                5.0 * ncols,
                4.6 * nrows,
            ),
        )

        axes = np.atleast_1d(
            axes
        ).ravel()

        for panel_ind, run_ind in enumerate(
            top_event_runs
        ):
            ax = axes[
                panel_ind
            ]

            switched = switch_mask[
                :,
                run_ind,
            ]

            eligible = eligible_mask[
                :,
                run_ind,
            ]

            ax.scatter(
                coords_scaled[
                    eligible,
                    0,
                ],
                coords_scaled[
                    eligible,
                    1,
                ],
                s=12,
                alpha=0.22,
                label="eligible",
            )

            ax.scatter(
                coords_scaled[
                    switched,
                    0,
                ],
                coords_scaled[
                    switched,
                    1,
                ],
                s=34,
                alpha=0.90,
                label="NV-→NV0",
            )

            ax.set_aspect(
                "equal",
                adjustable="box",
            )

            ax.invert_yaxis()

            ax.set_title(
                (
                    f"{dark_wait_by_run[run_ind]:g} s | "
                    f"run {global_local_run[run_ind]}\n"
                    f"loss={100*loss_fraction_by_run[run_ind]:.1f}% | "
                    f"z_loss={loss_z_same_wait[run_ind]:.1f} | "
                    f"z_spatial={spatial['short_spatial_z_by_run'][run_ind]:.1f}"
                ),
                fontsize=9,
            )

            ax.set_xlabel(
                distance_unit
            )

            ax.set_ylabel(
                distance_unit
            )

            ax.legend(
                fontsize=7,
            )

        for ax in axes[
            len(top_event_runs):
        ]:
            ax.axis("off")

        fig.suptitle(
            "Strongest wait-conditioned charge-loss candidates",
            fontsize=14,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.96,
            ]
        )

        figures[
            "top_event_maps"
        ] = fig

    # =====================================================================
    # Print summaries
    # =====================================================================

    if verbose:
        print(
            "\n"
            + "=" * 126
        )

        print(
            "DARK-TIME-DEPENDENT PARTICLE-MEMORY ANALYSIS"
        )

        print(
            "=" * 126
        )

        print(
            f"Files loaded: {len(datasets)}"
        )

        print(
            f"Total runs: {total_runs}"
        )

        print(
            f"NVs retained: {num_kept_nvs}"
        )

        print(
            f"Distance unit: {distance_unit}"
        )

        print(
            f"Short-range definition: 0-{short_range_max:g} {distance_unit}"
        )

        if loss_fit is not None:
            print(
                "Phenomenological loss fit:"
            )
            print(
                f"  L0   = {100*loss_fit['loss0']:.2f}%"
            )
            print(
                f"  Linf = {100*loss_fit['loss_inf']:.2f}%"
            )
            print(
                f"  tau  = {loss_fit['tau_s']:.1f} ± "
                f"{loss_fit['tau_s_err']:.1f} s"
            )

        print(
            "\nDARK-TIME SUMMARY"
        )

        print(
            "-" * 126
        )

        print(
            f"{'wait(s)':>8} "
            f"{'runs':>6} "
            f"{'loss med%':>10} "
            f"{'loss mean%':>11} "
            f"{'IQR%':>16} "
            f"{'ret med%':>10} "
            f"{'short enr':>10} "
            f"{'p_spatial':>10} "
            f"{'g_ind':>9} "
            f"{'p_ind':>9}"
        )

        for wait_ind, wait_s in enumerate(
            unique_waits
        ):
            print(
                f"{wait_s:8.0f} "
                f"{loss_stats['n'][wait_ind]:6d} "
                f"{loss_stats['median'][wait_ind]:10.3f} "
                f"{loss_stats['mean'][wait_ind]:11.3f} "
                f"{loss_stats['q25'][wait_ind]:7.2f}-"
                f"{loss_stats['q75'][wait_ind]:7.2f} "
                f"{retention_stats['median'][wait_ind]:10.3f} "
                f"{spatial['short_spatial_enrichment_by_wait'][wait_ind]:10.4f} "
                f"{spatial['short_p_spatial_by_wait'][wait_ind]:10.4f} "
                f"{spatial['short_g_independent_by_wait'][wait_ind]:9.4f} "
                f"{spatial['short_p_independent_by_wait'][wait_ind]:9.4f}"
            )

        print(
            "\nTOP WAIT-CONDITIONED CANDIDATE RUNS"
        )

        print(
            "-" * 126
        )

        print(
            f"{'global':>6} "
            f"{'file':>5} "
            f"{'run':>5} "
            f"{'wait(s)':>8} "
            f"{'eligible':>9} "
            f"{'lost':>6} "
            f"{'loss%':>8} "
            f"{'z_loss':>8} "
            f"{'p_loss':>8} "
            f"{'closepairs':>11} "
            f"{'z_spatial':>10} "
            f"{'p_spatial':>10}"
        )

        for ind in top_annotate:
            print(
                f"{ind:6d} "
                f"{global_file_index[ind]+1:5d} "
                f"{global_local_run[ind]:5d} "
                f"{dark_wait_by_run[ind]:8.0f} "
                f"{eligible_by_run[ind]:9d} "
                f"{switches_by_run[ind]:6d} "
                f"{100*loss_fraction_by_run[ind]:8.2f} "
                f"{loss_z_same_wait[ind]:8.2f} "
                f"{loss_empirical_p_same_wait[ind]:8.4f} "
                f"{spatial['short_pairs_observed_by_run'][ind]:11.0f} "
                f"{spatial['short_spatial_z_by_run'][ind]:10.2f} "
                f"{spatial['short_spatial_p_by_run'][ind]:10.4f}"
            )

    # =====================================================================
    # Package results
    # =====================================================================

    result = {
        "datasets":
            datasets,

        "global_runs":
            global_runs,

        "global_file_index":
            global_file_index,

        "global_local_run":
            global_local_run,

        "dark_wait_by_run":
            dark_wait_by_run,

        "unique_waits_s":
            unique_waits,

        "eligible_mask":
            eligible_mask,

        "switch_mask":
            switch_mask,

        "eligible_by_run":
            eligible_by_run,

        "switches_by_run":
            switches_by_run,

        "loss_fraction_by_run":
            loss_fraction_by_run,

        "retention_by_run":
            retention_by_run,

        "loss_z_same_wait":
            loss_z_same_wait,

        "loss_empirical_p_same_wait":
            loss_empirical_p_same_wait,

        "loss_stats_by_wait":
            loss_stats,

        "retention_stats_by_wait":
            retention_stats,

        "switch_stats_by_wait":
            switch_stats,

        "eligible_stats_by_wait":
            eligible_stats,

        "loss_fit":
            loss_fit,

        "coords_xy":
            master_coords,

        "coords_scaled":
            coords_scaled,

        "original_nv_inds":
            master_original_nv_inds,

        "distance_unit":
            distance_unit,

        "distance_bins":
            distance_bins,

        "distance_centers":
            distance_centers,

        "distance_matrix":
            distance_matrix,

        "short_range_max":
            short_range_max,

        "spatial":
            spatial,

        "top_event_runs":
            top_event_runs,
    }

    return result, figures


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()

    # -------------------------------------------------------------------------
    # Dark-time sweep
    # -------------------------------------------------------------------------

    FILE_STEMS = [
        "2026_07_24-21_43_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
        "2026_07_24-22_27_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
        "2026_07_24-23_44_20-qnami-nv0_2026_02_20-particle-memory-source_off_wait_30s-wait-30s",
        "2026_07_25-01_51_32-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
        "2026_07_25-05_57_38-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
        "2026_07_25-12_33_01-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
        "2026_07_25-21_07_06-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
        "2026_07_26-05_34_29-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
        "2026_07_26-18_11_11-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s",
        "2026_07_27-19_18_59-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    ]

    selected_waits_s = [
        0,
        10,
        30,
        60,
        180,
        300,
        600,
        1200,
        1800,
        3600,
    ]

    result, figures = analyze_charge_memory_vs_dark_time(
        FILE_STEMS,

        selected_waits_s=selected_waits_s,

        rep_initial=11,
        rep_final=12,

        # Start unfiltered. Add only independently justified bad NVs later.
        exclude_nv_inds=None,

        # Same threshold definition as the existing analysis.
        initial_margin_counts=0.0,
        final_margin_counts=0.0,

        # Camera calibration.
        um_per_pixel=0.43,

        # Spatial bins in micrometres.
        distance_bin_width=10.0,
        distance_max=None,

        # Predefined local-correlation scale.
        short_range_max=30.0,

        # 500 is useful for debugging.
        # 2000-5000 is better for final analysis.
        n_permutations=2000,

        top_n_event_maps=6,
        annotate_top_n=8,

        random_seed=12345,
        verbose=True,
    )

    kpl.show(block=True)



def plot_run_rep11_rep12_with_nv_circles(
    file_stem,
    run_ind=16,
    rep_initial=11,
    rep_final=12,
    exp_ind=0,
    step_ind=0,
    integration_radius_px=3.0,
    zoom_padding_px=10,
    show_nv_labels=False,
    estimate_drift=True,
    bright_margin_counts=5.0,
    drift_roi_radius_px=5,
    max_drift_nvs=200,
):
    """
    Plot raw images for one run at rep 11 and rep 12 with the SAME
    3-pixel integration circles over all NV positions.

    Panels
    ------
    1. rep 11 + fixed saved NV circles
    2. rep 12 + SAME fixed circles
    3. rep 12 + circles shifted by measured drift
    4. rep12 - rep11 difference image

    This is intended as a direct visual check of whether apparent charge loss
    could instead result from image/sample drift.

    Parameters
    ----------
    run_ind : int
        Python run index. run_ind=16 means counts/images[:, run 16, ...].
        If by "16th run" you mean human counting starting at 1, use run_ind=15.

    integration_radius_px : float
        Radius of the actual NV integration aperture. Set to 3 px here.
    """

    # =====================================================================
    # Load
    # =====================================================================

    raw_data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    nv_list = raw_data["nv_list"]

    # =====================================================================
    # PIXEL coordinates
    # =====================================================================

    coords_xy = []

    for nv_ind, nv in enumerate(nv_list):

        coords_dict = getattr(
            nv,
            "coords",
            None,
        )

        if not isinstance(coords_dict, dict):
            raise ValueError(
                f"NV {nv_ind} does not have a coords dictionary."
            )

        pixel_key = None

        for key in coords_dict.keys():

            if getattr(
                key,
                "name",
                None,
            ) == "PIXEL":

                pixel_key = key
                break

            key_string = str(key).upper()

            if (
                key_string == "PIXEL"
                or key_string.endswith(".PIXEL")
            ):
                pixel_key = key
                break

        if pixel_key is None:
            raise ValueError(
                f"Could not find PIXEL coordinate for NV {nv_ind}."
            )

        xy = np.asarray(
            coords_dict[pixel_key],
            dtype=float,
        ).ravel()

        coords_xy.append(
            [
                float(xy[0]),
                float(xy[1]),
            ]
        )

    coords_xy = np.asarray(
        coords_xy,
        dtype=float,
    )

    num_nvs = len(
        coords_xy
    )

    # =====================================================================
    # Images
    # =====================================================================

    if "img_arrays" not in raw_data:
        raise ValueError(
            "raw_data does not contain 'img_arrays'. "
            f"Available keys include: {list(raw_data.keys())}"
        )

    img_arrays = np.asarray(
        raw_data["img_arrays"],
        dtype=float,
    )

    print(
        "img_arrays shape:",
        img_arrays.shape,
    )

    # Expected:
    #
    # img_arrays[
    #     exp,
    #     run,
    #     step,
    #     rep,
    #     y,
    #     x,
    # ]
    #
    if img_arrays.ndim != 6:
        raise ValueError(
            "Expected img_arrays shape "
            "[exp, run, step, rep, y, x], "
            f"got {img_arrays.shape}"
        )

    num_runs = img_arrays.shape[1]
    num_reps = img_arrays.shape[3]

    if not (
        0 <= run_ind < num_runs
    ):
        raise ValueError(
            f"run_ind={run_ind} but dataset has "
            f"{num_runs} runs."
        )

    if (
        rep_initial >= num_reps
        or rep_final >= num_reps
    ):
        raise ValueError(
            f"Dataset only has {num_reps} reps."
        )

    img11 = np.asarray(
        img_arrays[
            exp_ind,
            run_ind,
            step_ind,
            rep_initial,
        ],
        dtype=float,
    )

    img12 = np.asarray(
        img_arrays[
            exp_ind,
            run_ind,
            step_ind,
            rep_final,
        ],
        dtype=float,
    )

    # =====================================================================
    # Counts + thresholds
    # =====================================================================

    counts_all = np.asarray(
        raw_data["counts"],
        dtype=float,
    )

    counts = counts_all[
        exp_ind,
        :,
        :,
        step_ind,
        :,
    ]

    counts11 = counts[
        :,
        run_ind,
        rep_initial,
    ]

    counts12 = counts[
        :,
        run_ind,
        rep_final,
    ]

    if "analysis_thresholds" in raw_data:

        thresholds = np.asarray(
            raw_data["analysis_thresholds"],
            dtype=float,
        )

    elif "thresholds" in raw_data:

        thresholds = np.asarray(
            raw_data["thresholds"],
            dtype=float,
        )

    else:

        thresholds = np.full(
            num_nvs,
            np.nan,
        )

    thresholds = np.squeeze(
        thresholds
    )

    # =====================================================================
    # Charge state categories
    # =====================================================================

    initial_nv_minus = (
        counts11 > thresholds
    )

    final_nv_minus = (
        counts12 > thresholds
    )

    lost = (
        initial_nv_minus
        & ~final_nv_minus
    )

    retained = (
        initial_nv_minus
        & final_nv_minus
    )

    gained = (
        ~initial_nv_minus
        & final_nv_minus
    )

    print()
    print("=" * 80)

    print(
        f"RUN {run_ind}"
    )

    print(
        f"rep {rep_initial} -> rep {rep_final}"
    )

    print(
        f"Initial NV- : {np.sum(initial_nv_minus)}"
    )

    print(
        f"Retained    : {np.sum(retained)}"
    )

    print(
        f"Lost        : {np.sum(lost)}"
    )

    print(
        f"Gained      : {np.sum(gained)}"
    )

    print("=" * 80)

    # =====================================================================
    # Estimate rep11 -> rep12 drift
    # =====================================================================

    dx_px = 0.0
    dy_px = 0.0
    drift_mag_px = 0.0
    drift_scatter_px = np.nan
    drift_result = None

    if estimate_drift:

        try:

            drift_result = (
                _estimate_run_drift_from_bright_nvs(
                    img11=img11,
                    img12=img12,
                    coords_xy=coords_xy,
                    counts11=counts11,
                    counts12=counts12,
                    thresholds=thresholds,
                    bright_margin_counts=bright_margin_counts,
                    roi_radius_px=drift_roi_radius_px,
                    max_reference_nvs=max_drift_nvs,
                )
            )

            dx_px = drift_result[
                "dx_px"
            ]

            dy_px = drift_result[
                "dy_px"
            ]

            drift_mag_px = drift_result[
                "magnitude_px"
            ]

            drift_scatter_px = drift_result[
                "scatter_px"
            ]

        except Exception as exc:

            print(
                "Drift estimation failed:",
                exc,
            )

    print()

    print(
        "Estimated rep11 -> rep12 drift:"
    )

    print(
        f"  dx = {dx_px:.4f} px"
    )

    print(
        f"  dy = {dy_px:.4f} px"
    )

    print(
        f"  |drift| = {drift_mag_px:.4f} px"
    )

    print(
        f"  scatter = {drift_scatter_px:.4f} px"
    )

    print(
        f"  integration radius = "
        f"{integration_radius_px:.1f} px"
    )

    if np.isfinite(
        drift_mag_px
    ):

        print(
            f"  drift / integration radius = "
            f"{drift_mag_px / integration_radius_px:.3f}"
        )

    # =====================================================================
    # Use common intensity range for rep11 and rep12
    # =====================================================================

    combined = np.concatenate(
        [
            img11.ravel(),
            img12.ravel(),
        ]
    )

    finite = combined[
        np.isfinite(
            combined
        )
    ]

    if len(finite) > 0:

        vmin = np.nanpercentile(
            finite,
            1,
        )

        vmax = np.nanpercentile(
            finite,
            99.8,
        )

    else:

        vmin = None
        vmax = None

    # =====================================================================
    # Zoom region containing NV array
    # =====================================================================

    x_min = max(
        0,
        np.floor(
            np.nanmin(
                coords_xy[:, 0]
            )
            - zoom_padding_px
        ),
    )

    x_max = min(
        img11.shape[1] - 1,
        np.ceil(
            np.nanmax(
                coords_xy[:, 0]
            )
            + zoom_padding_px
        ),
    )

    y_min = max(
        0,
        np.floor(
            np.nanmin(
                coords_xy[:, 1]
            )
            - zoom_padding_px
        ),
    )

    y_max = min(
        img11.shape[0] - 1,
        np.ceil(
            np.nanmax(
                coords_xy[:, 1]
            )
            + zoom_padding_px
        ),
    )

    # =====================================================================
    # Helper to draw circles
    # =====================================================================

    def add_nv_circles(
        ax,
        shift_x=0.0,
        shift_y=0.0,
    ):

        for nv_ind, (
            x0,
            y0,
        ) in enumerate(
            coords_xy
        ):

            circle = Circle(
                (
                    x0 + shift_x,
                    y0 + shift_y,
                ),
                radius=integration_radius_px,
                fill=False,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.75,
            )

            ax.add_patch(
                circle
            )

            if show_nv_labels:

                ax.text(
                    x0 + shift_x,
                    y0 + shift_y,
                    str(nv_ind),
                    fontsize=4,
                    ha="center",
                    va="center",
                )

    # =====================================================================
    # Main image comparison
    # =====================================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 13),
    )

    # ---------------------------------------------------------------------
    # Rep 11
    # ---------------------------------------------------------------------

    ax = axes[0, 0]

    ax.imshow(
        img11,
        origin="upper",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    add_nv_circles(
        ax,
        shift_x=0.0,
        shift_y=0.0,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_max,
        y_min,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        f"Run {run_ind} — rep {rep_initial}\n"
        "saved NV coordinates, r = 3 px"
    )

    ax.set_xlabel(
        "camera x (px)"
    )

    ax.set_ylabel(
        "camera y (px)"
    )

    # ---------------------------------------------------------------------
    # Rep 12 with EXACT SAME circles
    # ---------------------------------------------------------------------

    ax = axes[0, 1]

    ax.imshow(
        img12,
        origin="upper",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    add_nv_circles(
        ax,
        shift_x=0.0,
        shift_y=0.0,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_max,
        y_min,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        f"Run {run_ind} — rep {rep_final}\n"
        "SAME saved circles — inspect drift"
    )

    ax.set_xlabel(
        "camera x (px)"
    )

    ax.set_ylabel(
        "camera y (px)"
    )

    # ---------------------------------------------------------------------
    # Rep 12 with drift-corrected circles
    # ---------------------------------------------------------------------

    ax = axes[1, 0]

    ax.imshow(
        img12,
        origin="upper",
        interpolation="none",
        vmin=vmin,
        vmax=vmax,
    )

    add_nv_circles(
        ax,
        shift_x=dx_px,
        shift_y=dy_px,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_max,
        y_min,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        f"rep {rep_final} with measured shift\n"
        f"dx={dx_px:.3f}px, "
        f"dy={dy_px:.3f}px, "
        f"|d|={drift_mag_px:.3f}px"
    )

    ax.set_xlabel(
        "camera x (px)"
    )

    ax.set_ylabel(
        "camera y (px)"
    )

    # ---------------------------------------------------------------------
    # Difference image
    # ---------------------------------------------------------------------

    ax = axes[1, 1]

    difference = (
        img12
        - img11
    )

    diff_lim = np.nanpercentile(
        np.abs(
            difference
        ),
        99.5,
    )

    ax.imshow(
        difference,
        origin="upper",
        interpolation="none",
        vmin=-diff_lim,
        vmax=diff_lim,
        cmap="RdBu_r",
    )

    add_nv_circles(
        ax,
        shift_x=0.0,
        shift_y=0.0,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_max,
        y_min,
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_title(
        f"rep {rep_final} − rep {rep_initial}\n"
        "fixed r = 3 px circles"
    )

    ax.set_xlabel(
        "camera x (px)"
    )

    ax.set_ylabel(
        "camera y (px)"
    )

    fig.suptitle(
        (
            f"{file_stem}\n"
            f"Run {run_ind}: visual drift check"
        ),
        fontsize=13,
    )

    return {
        "figure":
            fig,

        "img11":
            img11,

        "img12":
            img12,

        "coords_xy":
            coords_xy,

        "counts11":
            counts11,

        "counts12":
            counts12,

        "thresholds":
            thresholds,

        "initial_nv_minus":
            initial_nv_minus,

        "retained":
            retained,

        "lost":
            lost,

        "gained":
            gained,

        "drift":
            drift_result,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()

    # FILE_STEM = "2026_07_27-19_18_59-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s"

    # run1_check = (
    #     plot_run_rep11_rep12_with_nv_circles(
    #         FILE_STEM,

    #         run_ind=1,

    #         rep_initial=11,
    #         rep_final=12,

    #         integration_radius_px=3.0,

    #         show_nv_labels=False,

    #         estimate_drift=True,
    #     )
    # )

    # kpl.show(block=True)
    # -------------------------------------------------------------------------
    # Dark-time sweep
    # -------------------------------------------------------------------------

    # FILE_STEMS = [
    #     "2026_07_24-21_43_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s",
    #     "2026_07_24-22_27_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_10s-wait-10s",
    #     "2026_07_24-23_44_20-qnami-nv0_2026_02_20-particle-memory-source_off_wait_30s-wait-30s",
    #     "2026_07_25-01_51_32-qnami-nv0_2026_02_20-particle-memory-source_off_wait_60s-wait-60s",
    #     "2026_07_25-05_57_38-qnami-nv0_2026_02_20-particle-memory-source_off_wait_180s-wait-180s",
    #     "2026_07_25-12_33_01-qnami-nv0_2026_02_20-particle-memory-source_off_wait_300s-wait-300s",
    #     "2026_07_25-21_07_06-qnami-nv0_2026_02_20-particle-memory-source_off_wait_600s-wait-600s",
    #     "2026_07_26-05_34_29-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1200s-wait-1200s",
    #     "2026_07_26-18_11_11-qnami-nv0_2026_02_20-particle-memory-source_off_wait_1800s-wait-1800s",
    #     "2026_07_27-19_18_59-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_13-11_33_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_14-02_16_30-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_14-14_19_56-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_15-02_23_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_15-14_26_42-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    # ]

    # FILE_STEMS = [
    #     "2026_08_13-11_33_24-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_14-02_16_30-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_14-14_19_56-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_15-02_23_19-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    #     "2026_08_15-14_26_42-qnami-nv0_2026_02_20-particle-memory-source_off_wait_3600s-wait-3600s",
    # ]

    
    ## 4000 measurement 11GB
    FILE_STEMS = [
    "2026_08_20-16_37_57-qnami-nv0_2026_02_20-particle-memory-source_off_wait_0s-wait-0s"
    ]

    selected_waits_s = [
        0,
        # 10,
        # 30,
        # 60,
        # 180,
        # 300,
        # 600,
        # 1200,
        # 1800,
        # 3600,
    ]
    result, figures = analyze_charge_memory_vs_dark_time(
        FILE_STEMS,

        selected_waits_s=selected_waits_s,

        rep_initial=11,
        rep_final=12,

        # Start unfiltered. Add only independently justified bad NVs later.
        exclude_nv_inds=None,

        # Same threshold definition as the existing analysis.
        initial_margin_counts=0.0,
        final_margin_counts=0.0,

        # Camera calibration.
        um_per_pixel=0.43,

        # Spatial bins in micrometres.
        distance_bin_width=10.0,
        distance_max=None,

        # Predefined local-correlation scale.
        short_range_max=30.0,

        # 500 is useful for debugging.
        # 2000-5000 is better for final analysis.
        n_permutations=2000,

        top_n_event_maps=6,
        annotate_top_n=8,

        random_seed=12345,
        verbose=True,
    )

    kpl.show(block=True)
