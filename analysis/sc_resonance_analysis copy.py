# -*- coding: utf-8 -*-
"""
Clean resonance histogram background-correction diagnostic.

Goal:
    Compare sig/ref raw count histograms before and after correcting
    branch/block-dependent low-count background offsets.

Dataset:
    2026_06_16-02_26_28-qnami-nv0_2026_02_20
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield
from analysis.sc_resonance_analysis import plot_nv_resonance



# =============================================================================
# Data splitting
# =============================================================================

def split_resonance_four_blocks(raw_data):
    """
    Split resonance sequence into four blocks:

        sig0, sig1, ref0, ref1

    Each block has shape:
        [nv, run, freq, rep]
    """

    counts = np.asarray(raw_data["counts"])[0]

    nv_list = raw_data["nv_list"]
    freqs = np.asarray(raw_data["freqs"])

    num_steps = int(raw_data["num_steps"])
    num_reps = int(raw_data["num_reps"])

    adj_num_steps = num_steps // 4

    sig0 = counts[:, :, 0:adj_num_steps, :]
    sig1 = counts[:, :, adj_num_steps:2 * adj_num_steps, :]
    ref0 = counts[:, :, 2 * adj_num_steps:3 * adj_num_steps, :]
    ref1 = counts[:, :, 3 * adj_num_steps:4 * adj_num_steps, :]

    # Guard in case freqs was saved as full length.
    if len(freqs) != adj_num_steps:
        freqs = freqs[:adj_num_steps]

    return nv_list, freqs, sig0, sig1, ref0, ref1, num_reps


def combine_sig_ref(sig0, sig1, ref0, ref1):
    """
    Combine the two signal blocks and two reference blocks.

    Signal:
        concatenate sig0 and sig1 along repetition axis.

    Reference:
        interleave ref0 and ref1 along repetition axis.
    """

    sig = np.concatenate([sig0, sig1], axis=3)

    num_nvs, num_runs, num_freqs, num_reps = ref0.shape

    ref = np.empty(
        (num_nvs, num_runs, num_freqs, 2 * num_reps),
        dtype=float,
    )

    ref[:, :, :, 0::2] = ref0
    ref[:, :, :, 1::2] = ref1

    return sig, ref


# =============================================================================
# Background / low-mode correction
# =============================================================================

def estimate_low_mode_level(block, q=0.15):
    """
    Estimate low-count/NV0-mode level for each NV and run.

    block shape:
        [nv, run, freq, rep]

    Output shape:
        [nv, run]

    This pools over all microwave frequencies and repetitions.
    That is intentional: we do not want to remove resonance contrast.
    """

    block = np.asarray(block, dtype=float)
    return np.nanquantile(block, q, axis=(2, 3))


def correct_four_blocks_by_low_mode(
    sig0,
    sig1,
    ref0,
    ref1,
    q=0.15,
    target_mode="all_blocks",
):
    """
    Correct block-dependent background offsets.

    For each NV and run, estimate the low-count level of each block:
        sig0_low, sig1_low, ref0_low, ref1_low

    Then align each block to a common target low-count level.

    target_mode:
        "all_blocks"  : target = median low level of sig0/sig1/ref0/ref1
        "ref_blocks"  : target = median low level of ref0/ref1 only

    Returns:
        corrected blocks and a diagnostics dictionary.
    """

    blocks = {
        "sig0": np.asarray(sig0, dtype=float),
        "sig1": np.asarray(sig1, dtype=float),
        "ref0": np.asarray(ref0, dtype=float),
        "ref1": np.asarray(ref1, dtype=float),
    }

    low = {
        name: estimate_low_mode_level(block, q=q)
        for name, block in blocks.items()
    }

    if target_mode == "all_blocks":
        target = np.nanmedian(
            np.stack([low["sig0"], low["sig1"], low["ref0"], low["ref1"]], axis=0),
            axis=0,
        )
    elif target_mode == "ref_blocks":
        target = np.nanmedian(
            np.stack([low["ref0"], low["ref1"]], axis=0),
            axis=0,
        )
    else:
        raise ValueError("target_mode must be 'all_blocks' or 'ref_blocks'.")

    corrected = {}
    offsets = {}

    for name, block in blocks.items():
        # Offset to subtract from this block.
        # Positive offset means this block's low mode was higher than target.
        offset = low[name] - target

        corrected[name] = block - offset[:, :, None, None]
        offsets[name] = offset

    diagnostics = {
        "low_raw": low,
        "target_low_mode": target,
        "offsets": offsets,
        "q": q,
        "target_mode": target_mode,
    }

    return (
        corrected["sig0"],
        corrected["sig1"],
        corrected["ref0"],
        corrected["ref1"],
        diagnostics,
    )


# =============================================================================
# Plotting diagnostics
# =============================================================================

def extract_counts_for_hist(block, nv_ind, freq_ind=None):
    """
    Extract counts for one NV.

    If freq_ind is None:
        pool over all frequencies/runs/reps.

    If freq_ind is int:
        use only that microwave frequency.
    """

    block = np.asarray(block, dtype=float)

    if freq_ind is None:
        vals = block[nv_ind].ravel()
    else:
        vals = block[nv_ind, :, freq_ind, :].ravel()

    vals = vals[np.isfinite(vals)]
    return vals


def plot_four_block_histograms_before_after(
    sig0_raw,
    sig1_raw,
    ref0_raw,
    ref1_raw,
    sig0_corr,
    sig1_corr,
    ref0_corr,
    ref1_corr,
    nv_ind,
    freq_ind=None,
    bins=80,
    density=True,
):
    """
    Plot raw and corrected histograms for sig0/sig1/ref0/ref1.
    """

    raw_blocks = {
        "sig0 raw": sig0_raw,
        "sig1 raw": sig1_raw,
        "ref0 raw": ref0_raw,
        "ref1 raw": ref1_raw,
    }

    corr_blocks = {
        "sig0 corrected": sig0_corr,
        "sig1 corrected": sig1_corr,
        "ref0 corrected": ref0_corr,
        "ref1 corrected": ref1_corr,
    }

    raw_vals = {
        name: extract_counts_for_hist(block, nv_ind, freq_ind)
        for name, block in raw_blocks.items()
    }

    corr_vals = {
        name: extract_counts_for_hist(block, nv_ind, freq_ind)
        for name, block in corr_blocks.items()
    }

    all_vals = list(raw_vals.values()) + list(corr_vals.values())
    all_concat = np.concatenate(all_vals)
    all_concat = all_concat[np.isfinite(all_concat)]

    bin_edges = np.linspace(
        np.nanpercentile(all_concat, 0.5),
        np.nanpercentile(all_concat, 99.5),
        bins + 1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    ax = axes[0]
    for name, vals in raw_vals.items():
        ax.hist(
            vals,
            bins=bin_edges,
            density=density,
            histtype="step",
            linewidth=1.8,
            label=f"{name}, q15={np.nanquantile(vals, 0.15):.1f}",
        )

    ax.set_title("Before correction")
    ax.set_xlabel("Integrated counts")
    ax.set_ylabel("Probability" if density else "Occurrences")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    ax = axes[1]
    for name, vals in corr_vals.items():
        ax.hist(
            vals,
            bins=bin_edges,
            density=density,
            histtype="step",
            linewidth=1.8,
            label=f"{name}, q15={np.nanquantile(vals, 0.15):.1f}",
        )

    ax.set_title("After correction")
    ax.set_xlabel("Integrated counts")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    if freq_ind is None:
        fig.suptitle(f"NV {nv_ind}: all frequencies pooled")
    else:
        fig.suptitle(f"NV {nv_ind}: frequency index {freq_ind}")

    fig.tight_layout()
    return fig


def summarize_sig_ref_low_mode_mismatch(diagnostics_raw_or_corr, label):
    """
    Print sig/ref low-mode mismatch statistics.

    mismatch = median(sig0,sig1 low mode) - median(ref0,ref1 low mode)
    """

    low = diagnostics_raw_or_corr["low_raw"]

    sig_low = np.nanmedian(
        np.stack([low["sig0"], low["sig1"]], axis=0),
        axis=0,
    )

    ref_low = np.nanmedian(
        np.stack([low["ref0"], low["ref1"]], axis=0),
        axis=0,
    )

    mismatch = sig_low - ref_low
    mismatch_flat = mismatch[np.isfinite(mismatch)]

    print(f"\n=== {label}: sig/ref low-mode mismatch ===")
    print("Median sig_low - ref_low:", np.nanmedian(mismatch_flat))
    print("Mean sig_low - ref_low:", np.nanmean(mismatch_flat))
    print("10/90 percentile:", np.nanpercentile(mismatch_flat, [10, 90]))

    return mismatch


def make_low_mode_diagnostics_for_blocks(sig0, sig1, ref0, ref1, q=0.15):
    """
    Convenience wrapper to compute low-mode diagnostics without correcting.
    """

    low = {
        "sig0": estimate_low_mode_level(sig0, q=q),
        "sig1": estimate_low_mode_level(sig1, q=q),
        "ref0": estimate_low_mode_level(ref0, q=q),
        "ref1": estimate_low_mode_level(ref1, q=q),
    }

    return {
        "low_raw": low,
        "q": q,
    }


def plot_mismatch_histogram(mismatch_raw, mismatch_corr, bins=80):
    """
    Plot distribution of sig/ref low-mode mismatch before and after correction.
    """

    raw = mismatch_raw[np.isfinite(mismatch_raw)].ravel()
    corr = mismatch_corr[np.isfinite(mismatch_corr)].ravel()

    all_vals = np.concatenate([raw, corr])
    bin_edges = np.linspace(
        np.nanpercentile(all_vals, 0.5),
        np.nanpercentile(all_vals, 99.5),
        bins + 1,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        raw,
        bins=bin_edges,
        histtype="step",
        linewidth=2,
        density=True,
        label="before correction",
    )

    ax.hist(
        corr,
        bins=bin_edges,
        histtype="step",
        linewidth=2,
        density=True,
        label="after correction",
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("sig low-mode - ref low-mode counts")
    ax.set_ylabel("Probability")
    ax.set_title("Sig/ref NV0-mode mismatch before and after correction")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    return fig


# =============================================================================
# Optional: resonance plotting wrapper
# =============================================================================

def run_process_counts_quick_check(nv_list, sig_counts, ref_counts):
    """
    Quick check using your standard widefield processing.
    """

    avg_counts, avg_counts_ste = widefield.process_counts(
        nv_list,
        sig_counts,
        ref_counts,
        threshold=True,
    )

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)

    print("\n=== processed counts quick check ===")
    print("avg_counts shape:", np.shape(avg_counts))
    print("avg_counts median:", np.nanmedian(avg_counts))
    print("avg_snr median:", np.nanmedian(avg_snr))

    return avg_counts, avg_counts_ste, avg_snr, avg_snr_ste


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()
    # =============================================================================
    # User settings
    # =============================================================================

    # FILE_STEM = "2026_06_16-02_26_28-qnami-nv0_2026_02_20"
    FILE_STEM = "2026_06_16-21_08_58-qnami-nv0_2026_02_20"
    # FILE_STEM = "2026_06_17-07_16_02-qnami-nv0_2026_02_20"

    NV_CHECK = 10          # NV/pillar index to inspect
    FREQ_CHECK = None      # None = pool all microwave freqs; or set e.g. 5
    LOW_MODE_Q = 0.15      # lower quantile used as NV0/background proxy

    DO_PLOT_RESONANCE_RAW = False
    DO_PLOT_RESONANCE_CORRECTED = True
    
    raw_data = dm.get_raw_data(
        file_stem=FILE_STEM,
        load_npz=True,
        use_cache=True,
    )

    nv_list, freqs, sig0, sig1, ref0, ref1, num_reps = split_resonance_four_blocks(
        raw_data
    )

    print("Loaded:", FILE_STEM)
    print("Number of NVs:", len(nv_list))
    print("Number of frequencies:", len(freqs))
    print("num_reps:", num_reps)
    print("sig0 shape:", sig0.shape)
    print("ref0 shape:", ref0.shape)

    # Raw combined signal/reference.
    sig_raw, ref_raw = combine_sig_ref(sig0, sig1, ref0, ref1)

    # Raw mismatch diagnostic.
    raw_diag = make_low_mode_diagnostics_for_blocks(
        sig0,
        sig1,
        ref0,
        ref1,
        q=LOW_MODE_Q,
    )

    mismatch_raw = summarize_sig_ref_low_mode_mismatch(
        raw_diag,
        label="Before correction",
    )

    # Correct four blocks.
    sig0_corr, sig1_corr, ref0_corr, ref1_corr, corr_diag = correct_four_blocks_by_low_mode(
        sig0,
        sig1,
        ref0,
        ref1,
        q=LOW_MODE_Q,
        target_mode="all_blocks",
    )

    # Corrected combined signal/reference.
    sig_corr, ref_corr = combine_sig_ref(
        sig0_corr,
        sig1_corr,
        ref0_corr,
        ref1_corr,
    )

    # Corrected mismatch diagnostic.
    corr_check_diag = make_low_mode_diagnostics_for_blocks(
        sig0_corr,
        sig1_corr,
        ref0_corr,
        ref1_corr,
        q=LOW_MODE_Q,
    )

    mismatch_corr = summarize_sig_ref_low_mode_mismatch(
        corr_check_diag,
        label="After correction",
    )

    # Plot one NV/pillar histogram before/after.
    plot_four_block_histograms_before_after(
        sig0,
        sig1,
        ref0,
        ref1,
        sig0_corr,
        sig1_corr,
        ref0_corr,
        ref1_corr,
        nv_ind=NV_CHECK,
        freq_ind=FREQ_CHECK,
        bins=80,
        density=True,
    )

    # Plot mismatch distribution for all NVs/runs.
    plot_mismatch_histogram(
        mismatch_raw,
        mismatch_corr,
        bins=80,
    )

    # Quick processed-count check before/after.
    print("\nRaw counts:")
    run_process_counts_quick_check(nv_list, sig_raw, ref_raw)

    print("\nCorrected counts:")
    run_process_counts_quick_check(nv_list, sig_corr, ref_corr)

    # Optional: run your existing resonance plotting/fitting function.
    # This assumes plot_nv_resonance(...) is defined above in your file.
    if DO_PLOT_RESONANCE_RAW:
        plot_nv_resonance(
            nv_list,
            freqs,
            sig_raw,
            ref_raw,
            file_id=FILE_STEM + "_raw",
            num_cols=8,
        )

    if DO_PLOT_RESONANCE_CORRECTED:
        plot_nv_resonance(
            nv_list,
            freqs,
            sig_corr,
            ref_corr,
            file_id=FILE_STEM + "_background_corrected",
            num_cols=8,
        )

    kpl.show(block=True)