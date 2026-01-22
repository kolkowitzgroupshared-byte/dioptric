# dm_lockin_fft_corr_clean.py
# Clean minimal utilities for:
#   (1) build complex lock-in series s_nv[t] = I + iQ from counts
#   (2) trim / drop bad runs + bad shots
#   (3) detrend (per-run mean), regress out global mean(I,Q), whiten
#   (4) correlate across NV index n (time-domain covariance/coherence)
#   (5) FFT + frequency-domain cross-NV coherence in a chosen band
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import widefield
# ----------------------------
# 1) Build complex lock-in s = I + iQ
# ----------------------------
def lockin_iq_from_counts(counts, eps=1e-12, return_masks=False):
    """
    counts: array (4, M, R, S, P) with ordering [Ip, Im, Qp, Qm]
    Returns:
      I, Q, s with shape (M, N) where N=R*S*P
      (optional) mI, mQ with shape (M, N)
    """
    counts = np.asarray(counts)
    if counts.dtype == object:
        counts = counts.astype(float)

    assert counts.ndim == 5 and counts.shape[0] == 4, f"counts shape must be (4,M,R,S,P), got {counts.shape}"
    Ip = counts[0].astype(float)
    Im = counts[1].astype(float)
    Qp = counts[2].astype(float)
    Qm = counts[3].astype(float)

    M = Ip.shape[0]
    Ip_f = Ip.reshape(M, -1)
    Im_f = Im.reshape(M, -1)
    Qp_f = Qp.reshape(M, -1)
    Qm_f = Qm.reshape(M, -1)

    denI = Ip_f + Im_f
    denQ = Qp_f + Qm_f

    mI = np.isfinite(Ip_f) & np.isfinite(Im_f) & np.isfinite(denI) & (denI > 0)
    mQ = np.isfinite(Qp_f) & np.isfinite(Qm_f) & np.isfinite(denQ) & (denQ > 0)

    I = np.full_like(denI, np.nan, dtype=float)
    Q = np.full_like(denQ, np.nan, dtype=float)

    I[mI] = (Ip_f[mI] - Im_f[mI]) / (denI[mI] + eps)
    Q[mQ] = (Qp_f[mQ] - Qm_f[mQ]) / (denQ[mQ] + eps)

    s = I + 1j * Q
    if return_masks:
        return I, Q, s, mI, mQ
    return I, Q, s


# ----------------------------
# 2) Bad-run / bad-shot handling
# ----------------------------
def _shot_bad_fraction(counts, eps=1e-12):
    """
    counts: (4,M,R,S,P)
    Returns frac_bad_per_shot: (N,) where N=R*S*P,
    where "bad" means invalid I or invalid Q for that NV+shot.
    """
    _, _, s, mI, mQ = lockin_iq_from_counts(counts, eps=eps, return_masks=True)
    bad = (~mI) | (~mQ)  # (M,N)
    return bad.mean(axis=0)  # (N,)


def find_bad_runs(counts, global_bad_thresh=0.9, eps=1e-12):
    """
    A shot is "global-bad" if >global_bad_thresh fraction of NVs invalid.
    A run is "bad" if >global_bad_thresh fraction of its shots are global-bad.
    Returns bad_run_indices (np.ndarray).
    """
    counts = np.asarray(counts)
    _, M, R, S, P = counts.shape
    frac_bad_shot = _shot_bad_fraction(counts, eps=eps)  # (N,)
    global_bad = (frac_bad_shot > global_bad_thresh).reshape(R, S, P)
    frac_global_bad_per_run = global_bad.mean(axis=(1, 2))  # (R,)
    bad_runs = np.where(frac_global_bad_per_run > global_bad_thresh)[0]
    return bad_runs


def trim_trailing_bad_runs(counts, global_bad_thresh=0.9, eps=1e-12):
    """
    If bad runs form a trailing contiguous block, cut them off.
    Returns (counts_trim, R_keep).
    """
    counts = np.asarray(counts)
    _, _, R, _, _ = counts.shape
    bad_runs = find_bad_runs(counts, global_bad_thresh=global_bad_thresh, eps=eps)
    if bad_runs.size == 0:
        return counts, R
    r0 = int(bad_runs[0])
    trailing = np.all(bad_runs == np.arange(r0, R))
    if trailing:
        return counts[:, :, :r0, :, :], r0
    return counts, R


def drop_bad_runs_anywhere(counts, global_bad_thresh=0.9, eps=1e-12):
    """
    Drops any run classified as bad (not just trailing).
    Preserves run order.
    Returns (counts_keep, keep_runs).
    """
    counts = np.asarray(counts)
    _, _, R, _, _ = counts.shape
    bad_runs = find_bad_runs(counts, global_bad_thresh=global_bad_thresh, eps=eps)
    if bad_runs.size == 0:
        keep = np.arange(R, dtype=int)
        return counts, keep
    keep = np.setdiff1d(np.arange(R), bad_runs)
    return counts[:, :, keep, :, :], keep


def drop_shots_with_any_nan(s_mn):
    """
    Keep only shots where all NVs have finite complex s.
    s_mn: (M,N)
    Returns s_keep, keep_mask (N,)
    """
    s_mn = np.asarray(s_mn)
    keep = np.all(np.isfinite(s_mn), axis=0)
    return s_mn[:, keep], keep


# ----------------------------
# 3) Detrend / regress / whiten
# ----------------------------
def detrend_per_run_mean(x_mn, R, S, P):
    """
    Subtract per-NV, per-run mean over shots within that run.
    x_mn: (M, N), N=R*S*P
    """
    M, N = x_mn.shape
    assert N == R * S * P, "x_mn length must match R*S*P"
    x = x_mn.reshape(M, R, S, P)
    mu = np.mean(x, axis=(2, 3), keepdims=True)  # (M,R,1,1)
    return (x - mu).reshape(M, N)


def regress_out_global_IQ(I_mn, Q_mn, use_Q=True):
    """
    Regress out global mean across NVs per shot:
      gI[t]=mean_n I[n,t], optionally gQ[t]
    Returns (I_res, Q_res, reg_dict)
    """
    I = np.asarray(I_mn, float)
    Q = np.asarray(Q_mn, float)
    M, N = I.shape

    gI = np.mean(I, axis=0)
    gQ = np.mean(Q, axis=0)

    if use_Q:
        X = np.vstack([gI, gQ])  # (K,N)
    else:
        X = np.vstack([gI])      # (1,N)

    X = X - X.mean(axis=1, keepdims=True)
    XXT_inv = np.linalg.pinv(X @ X.T)
    P = XXT_inv @ X  # (K,N)

    betaI = I @ P.T
    betaQ = Q @ P.T

    I_res = I - (betaI @ X)
    Q_res = Q - (betaQ @ X)

    return I_res, Q_res, {"gI": gI, "gQ": gQ, "betaI": betaI, "betaQ": betaQ}


def whiten_per_nv(s_mn, eps=1e-12):
    """
    Mean-remove per NV, then scale so mean power per NV is 1:
      z = (s - mean(s)) / sqrt(mean(|s-mean|^2))
    """
    s = np.asarray(s_mn, np.complex128)
    mu = np.mean(s, axis=1, keepdims=True)
    x = s - mu
    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    z = x / np.sqrt(np.maximum(p, eps))
    return z


# ----------------------------
# 4) Correlate across NV index n (time-domain)
# ----------------------------
def coherence_matrix_time(z_mn):
    """
    z_mn: (M,N) whitened (diag ~ 1)
    Returns C: (M,M) complex, Hermitian, diag ~ 1
    """
    z = np.asarray(z_mn, np.complex128)
    M, N = z.shape
    C = (z @ z.conj().T) / float(max(N, 1))
    return 0.5 * (C + C.conj().T)


def eig_hermitian(C):
    """
    Returns evals (descending), evecs columns aligned to evals.
    """
    C = 0.5 * (C + C.conj().T)
    w, V = np.linalg.eigh(C)  # ascending
    order = np.argsort(np.real(w))[::-1]
    return np.real(w[order]), V[:, order]


def common_mode_timeseries(z_mn, v_m):
    """
    a[t] = v^H z[:,t]
    """
    return (v_m.conj().T @ z_mn).astype(np.complex128)


# ----------------------------
# 5) FFT + frequency-domain coherence across NVs
# ----------------------------
def fft_complex(x, dt=1.0, window="hann"):
    """
    x: (N,) complex or real time series
    dt: sampling interval (seconds). If unknown, dt=1 => units are cycles/shot.
    window: 'hann' or None
    Returns (f_pos, X_pos) for positive frequencies only.
    """
    x = np.asarray(x, np.complex128)
    N = x.size
    if window == "hann":
        w = np.hanning(N)
        xw = x * w
    else:
        w = None
        xw = x

    X = np.fft.fft(xw)
    f = np.fft.fftfreq(N, d=dt)

    pos = f >= 0
    f_pos = f[pos]
    X_pos = X[pos]
    return f_pos, X_pos


def fft_all_nvs(z_mn, dt=1.0, window="hann"):
    """
    z_mn: (M,N) complex
    Returns:
      f_pos: (F,)
      Zf: (M,F) complex FFT values (positive freqs)
    """
    z = np.asarray(z_mn, np.complex128)
    M, N = z.shape
    if window == "hann":
        w = np.hanning(N)[None, :]
        zw = z * w
    else:
        zw = z

    Z = np.fft.fft(zw, axis=1)
    f = np.fft.fftfreq(N, d=dt)
    pos = f >= 0
    return f[pos], Z[:, pos]


def spectral_covariance(Zf_mf, f, fmin=None, fmax=None):
    """
    Build a cross-NV covariance matrix by averaging over frequency bins in [fmin,fmax].
    Zf_mf: (M,F) complex FFT values
    f: (F,) positive frequencies
    Returns Cspec: (M,M) complex (Hermitian)
    """
    Zf = np.asarray(Zf_mf, np.complex128)
    f = np.asarray(f, float)

    sel = np.ones_like(f, dtype=bool)
    if fmin is not None:
        sel &= (f >= float(fmin))
    if fmax is not None:
        sel &= (f <= float(fmax))

    Zb = Zf[:, sel]
    if Zb.shape[1] < 1:
        raise ValueError("No FFT bins selected in the requested band.")

    C = (Zb @ Zb.conj().T) / float(Zb.shape[1])
    return 0.5 * (C + C.conj().T)


def coherence_from_cov(C, eps=1e-18):
    """
    Normalize C -> coh_ij = C_ij / sqrt(C_ii C_jj)
    """
    C = np.asarray(C, np.complex128)
    p = np.real(np.diag(C))
    p = np.maximum(p, eps)
    denom = np.sqrt(p[:, None] * p[None, :])
    return C / denom


# ----------------------------
# One-shot pipeline (minimal)
# ----------------------------
def analyze_counts_fft_corr(
    counts,
    dt=1.0,
    global_bad_thresh=0.9,
    trim_trailing=True,
    drop_all_bad_runs=False,
    detrend=True,
    regress_global=True,
    regress_use_Q=True,
    fband=None,  # (fmin,fmax) or None
    eps=1e-12,
):
    """
    counts: (4,M,R,S,P) with ordering [Ip, Im, Qp, Qm]
    dt: seconds per shot (or dt=1 for cycles/shot)
    fband: None or tuple (fmin,fmax) for spectral coherence matrix

    Returns dict with:
      - z: (M,N) whitened complex lock-in series
      - C_time: (M,M) time-domain coherence/covariance across NV index
      - evals_time, v1_time, a_time: eigendecomp + top-mode timeseries
      - f_pos, Zf: per-NV FFT
      - optional: C_spec, coh_spec, evals_spec, v1_spec (band-averaged)
      - diagnostics: bad-run stats, keep masks, regress info, intermediates
    """
    counts = np.asarray(counts)

    # ----------------------------
    # A) Pre-diagnostics on raw counts (before trimming/dropping)
    # ----------------------------
    frac_bad_shot0 = _shot_bad_fraction(counts, eps=eps)  # (N_total,) where N_total=R*S*P
    _, M0, R0, S0, P0 = counts.shape
    global_bad0 = (frac_bad_shot0 > global_bad_thresh).reshape(R0, S0, P0)
    frac_global_bad_per_run0 = global_bad0.mean(axis=(1, 2))  # (R0,)

    # ----------------------------
    # B) Optionally trim/drop runs
    # ----------------------------
    keep_runs = None
    if trim_trailing:
        counts, R_keep = trim_trailing_bad_runs(counts, global_bad_thresh=global_bad_thresh, eps=eps)
        # if we trimmed trailing runs, keep_runs is the prefix
        keep_runs = np.arange(R_keep, dtype=int)

    if drop_all_bad_runs:
        counts, keep_runs2 = drop_bad_runs_anywhere(counts, global_bad_thresh=global_bad_thresh, eps=eps)
        keep_runs = keep_runs2  # overrides (or refines) keep_runs

    # updated dims after run filtering
    _, M, R, S, P = counts.shape

    # ----------------------------
    # C) Build I0,Q0 on the FULL (R,S,P) grid, then detrend/regress,
    #    then drop bad shots. (This avoids reshape issues.)
    # ----------------------------
    I0, Q0, _, mI, mQ = lockin_iq_from_counts(counts, eps=eps, return_masks=True)  # (M, R*S*P)

    # Detrend per-run mean (requires exact R,S,P structure)
    if detrend:
        I0 = detrend_per_run_mean(I0, R, S, P)
        Q0 = detrend_per_run_mean(Q0, R, S, P)

    # Regress out global mean across NVs per shot
    if regress_global:
        I0, Q0, reg = regress_out_global_IQ(I0, Q0, use_Q=regress_use_Q)
    else:
        reg = None

    # Complex series
    s0_full = I0 + 1j * Q0

    # Drop any shots that have any NaN across NVs
    s0, keep_shots = drop_shots_with_any_nan(s0_full)  # keep_shots length = R*S*P (after run dropping)
    # For convenience, also keep the post-processed I0,Q0 after shot dropping
    I0_keep = np.real(s0)
    Q0_keep = np.imag(s0)

    # ----------------------------
    # D) Whiten per NV
    # ----------------------------
    z = whiten_per_nv(s0, eps=eps)  # (M, Nshots)

    # ----------------------------
    # E) Time-domain coherence across NV index
    # ----------------------------
    C_time = coherence_matrix_time(z)
    evals_t, evecs_t = eig_hermitian(C_time)
    v1_t = evecs_t[:, 0]
    a_t = common_mode_timeseries(z, v1_t)

    # ----------------------------
    # F) FFT per NV (positive freqs)
    # ----------------------------
    f_pos, Zf = fft_all_nvs(z, dt=dt, window="hann")

    # ----------------------------
    # G) Optional spectral band coherence
    # ----------------------------
    out = {
        # shapes / bookkeeping
        "counts_used_shape": counts.shape,
        "M": M, "R": R, "S": S, "P": P,
        "Nshots": z.shape[1],
        "dt": dt,

        # keep masks
        "keep_runs": keep_runs,       # None or array of kept run indices (relative to original)
        "keep_shots": keep_shots,     # boolean mask relative to flattened shots of kept runs

        # raw diagnostics (before trimming/dropping)
        "frac_bad_shot0": frac_bad_shot0,
        "frac_global_bad_per_run0": frac_global_bad_per_run0,

        # intermediates (post-detrend/regress; BEFORE whitening)
        "I0_full": I0,                # (M, R*S*P) after run filtering, before shot drop
        "Q0_full": Q0,
        "s0_full": s0_full,
        "I0": I0_keep,                # (M, Nshots) after shot drop
        "Q0": Q0_keep,
        "s0": s0,                     # (M, Nshots)
        "reg": reg,

        # main outputs
        "z": z,
        "C_time": C_time,
        "evals_time": evals_t,
        "v1_time": v1_t,
        "a_time": a_t,

        # FFT outputs
        "f_pos": f_pos,
        "Zf": Zf,
    }

    if fband is not None:
        fmin, fmax = fband
        C_spec = spectral_covariance(Zf, f_pos, fmin=fmin, fmax=fmax)
        coh_spec = coherence_from_cov(C_spec)
        evals_s, evecs_s = eig_hermitian(coh_spec)
        out.update({
            "fband": (fmin, fmax),
            "C_spec": C_spec,
            "coh_spec": coh_spec,
            "evals_spec": evals_s,
            "v1_spec": evecs_s[:, 0],
        })

    return out


def plot_bad_run_stats(out, global_bad_thresh=0.9):
    frac_run = out.get("frac_global_bad_per_run0", None)
    if frac_run is None:
        return
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(frac_run, "o-")
    ax.axhline(global_bad_thresh, ls="--")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Frac(global-bad shots)")
    ax.set_title("Bad-run diagnostic")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_lockin_hist(out, max_points=200000):
    I0 = out.get("I0", None)
    Q0 = out.get("Q0", None)
    if I0 is None or Q0 is None:
        return

    # downsample for speed
    I = I0.ravel()
    Q = Q0.ravel()
    m = np.isfinite(I) & np.isfinite(Q)
    I = I[m]; Q = Q[m]
    if I.size > max_points:
        idx = np.random.choice(I.size, size=max_points, replace=False)
        I = I[idx]; Q = Q[idx]

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(I, Q, ".", ms=1, alpha=0.3)
    ax.set_xlabel("I (after detrend/regress)")
    ax.set_ylabel("Q (after detrend/regress)")
    ax.set_title("IQ cloud (sanity check)")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_example_timeseries(out, nv_indices=(0), nshots=6000):
    z = out["z"]
    N = z.shape[1]
    nshots = min(nshots, N)
    t = np.arange(nshots)

    fig, ax = plt.subplots(figsize=(7,4))
    for i in nv_indices:
        if i < 0 or i >= z.shape[0]:
            continue
        ax.plot(t, np.real(z[i,:nshots]), label=f"NV {i} Re(z)")
    ax.set_xlabel("Shot index")
    ax.set_ylabel("Re(z) (whitened)")
    ax.set_title("Example whitened traces")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()



def make_orientation_permutation(M, ORI_11m1, ORI_m111, keep_rest=True):
    """
    Returns:
      perm: indices length M (new order)
      groups: dict with group slices for plotting boundaries
    """
    ORI_11m1 = [i for i in ORI_11m1 if 0 <= i < M]
    ORI_m111 = [i for i in ORI_m111 if 0 <= i < M]

    # remove overlaps (just in case)
    set11 = set(ORI_11m1)
    setm1 = [i for i in ORI_m111 if i not in set11]

    perm = ORI_11m1 + setm1

    if keep_rest:
        rest = [i for i in range(M) if i not in set(perm)]
        perm = perm + rest

    perm = np.array(perm, dtype=int)

    n1 = len(ORI_11m1)
    n2 = len(setm1)
    groups = {
        "ORI_11m1": (0, n1),
        "ORI_m111": (n1, n1+n2),
        "rest": (n1+n2, len(perm))
    }
    return perm, groups


def plot_matrix_with_boundaries(C, perm, groups, title="C", abs_phase="abs",
                                prc=(1,99), diag_color="0.85"):
    """
    abs_phase: "abs" or "phase"
    """
    C = np.asarray(C)
    Cp = C[np.ix_(perm, perm)]

    # choose data to show
    if abs_phase == "abs":
        A = np.abs(Cp)
        ttl = title + " |abs|"
    else:
        A = np.angle(Cp)
        ttl = title + " phase"

    # mask diagonal (display gray) + off-diag autoscale
    mask = np.eye(A.shape[0], dtype=bool)
    Am = np.ma.array(A, mask=mask)

    # off-diagonal limits
    A_tmp = np.array(A, float, copy=True)
    np.fill_diagonal(A_tmp, np.nan)
    vmin, vmax = np.nanpercentile(A_tmp, prc)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(diag_color)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(ttl + " (sorted by orientation)")
    ax.set_xlabel("NV index (sorted)")
    ax.set_ylabel("NV index (sorted)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # draw boundaries + labels
    for name, (a, b) in groups.items():
        if name == "rest" and (b - a) == 0:
            continue
        if a > 0:
            ax.axhline(a - 0.5, color="w", lw=1.2)
            ax.axvline(a - 0.5, color="w", lw=1.2)

    # label group centers on axes
    for name, (a, b) in groups.items():
        if (b - a) <= 0:
            continue
        c = 0.5 * (a + b - 1)
        ax.text(-0.02, c, name, va="center", ha="right", transform=ax.get_yaxis_transform())
        ax.text(c, -0.03, name, va="top", ha="center", transform=ax.get_xaxis_transform())

    plt.tight_layout()
    return fig, ax, Cp

def _offdiag_limits(A, prc=(1, 99)):
    A = np.array(A, float, copy=True)
    np.fill_diagonal(A, np.nan)
    vmin, vmax = np.nanpercentile(A, prc)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(A), np.nanmax(A)
    return vmin, vmax

def plot_matrix(
    C,
    title="Matrix",
    show_phase=True,
    prc=(1, 99),
    diag_color="lightgray",   # <- diagonal displayed as this color
):
    C = np.asarray(C)

    def _imshow_with_gray_diag(ax, A, vmin=None, vmax=None):
        A = np.asarray(A)
        mask = np.eye(A.shape[0], dtype=bool)
        Am = np.ma.array(A, mask=mask)  # mask diagonal

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(diag_color)        # masked values -> gray

        im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
        return im

    # --- |C| ---
    A = np.abs(C)
    vmin, vmax = _offdiag_limits(A, prc=prc)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = _imshow_with_gray_diag(ax, A, vmin=vmin, vmax=vmax)
    ax.set_title(title + " |abs| (diag gray, off-diag autoscale)")
    ax.set_xlabel("NV index")
    ax.set_ylabel("NV index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # --- phase ---
    if show_phase:
        P = np.angle(C)
        vmin, vmax = _offdiag_limits(P, prc=prc)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = _imshow_with_gray_diag(ax, P, vmin=vmin, vmax=vmax)
        ax.set_title(title + " phase (diag gray, off-diag autoscale)")
        ax.set_xlabel("NV index")
        ax.set_ylabel("NV index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

def plot_eigs(evals, title="Eigenvalues"):
    evals = np.asarray(evals, float)
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(evals, "o-")
    ax.set_xlabel("mode index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_common_mode(out, dt=1.0, fmax=None):
    a = out["a_time"]
    N = a.size
    t = np.arange(N) * dt

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(t, np.real(a), lw=1, label="Re(a)")
    ax.plot(t, np.imag(a), lw=1, label="Im(a)")
    ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
    ax.set_ylabel("a(t)")
    ax.set_title("Top-mode time series a(t)")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend()
    plt.tight_layout()

    # spectrum of a(t)
    A = np.fft.rfft(a * np.hanning(N))
    f = np.fft.rfftfreq(N, d=dt)
    psd = np.abs(A)**2

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(f, psd)
    ax.set_xlabel("frequency (Hz)" if dt != 1.0 else "cycles/shot")
    ax.set_ylabel("|A(f)|^2")
    ax.set_title("Spectrum of a(t)")
    ax.grid(True, ls="--", lw=0.5)
    if fmax is not None:
        ax.set_xlim(0, fmax)
    plt.tight_layout()

def plot_nv_spectrum_median(out, fmax=None):
    f = out["f_pos"]
    Zf = out["Zf"]
    psd_med = np.median(np.abs(Zf)**2, axis=0)

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(f, psd_med)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("median |Z(f)|^2 across NVs")
    ax.set_title("Median per-NV spectrum")
    ax.grid(True, ls="--", lw=0.5)
    if fmax is not None:
        ax.set_xlim(0, fmax)
    plt.tight_layout()

def make_all_plots(out, dt=1.0, global_bad_thresh=0.9, fmax=None):
    plot_bad_run_stats(out, global_bad_thresh=global_bad_thresh)
    plot_lockin_hist(out)
    plot_example_timeseries(out, nv_indices=(0,1,2,3), nshots=600)

    plot_matrix(out["C_time"], title="Time-domain coherence (NV×NV)", show_phase=False)
    plot_eigs(out["evals_time"], title="Eigenvalues of time-domain coherence")

    plot_common_mode(out, dt=dt, fmax=fmax)
    plot_nv_spectrum_median(out, fmax=fmax)

    if "coh_spec" in out:
        plot_matrix(out["coh_spec"], title="Spectral coherence (band-averaged)", show_phase=False)
        plot_eigs(out["evals_spec"], title="Eigenvalues of spectral coherence")



def block_contrast_abs(C, g1, g2):
    """Contrast = mean(|C| within groups) - mean(|C| cross), excluding diagonal."""
    C = np.asarray(C)
    idx1 = np.array(g1, int); idx2 = np.array(g2, int)

    C11 = np.abs(C[np.ix_(idx1, idx1)])
    C22 = np.abs(C[np.ix_(idx2, idx2)])
    C12 = np.abs(C[np.ix_(idx1, idx2)])

    # drop diagonal for within blocks
    np.fill_diagonal(C11, np.nan)
    np.fill_diagonal(C22, np.nan)

    within = np.nanmean(np.r_[C11.ravel(), C22.ravel()])
    cross  = np.nanmean(C12.ravel())
    return float(within - cross), float(within), float(cross)

def perm_test_orientation(C, ORI_11m1, ORI_m111, nperm=2000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.asarray(C)
    M = C.shape[0]

    # make clean groups with in-range indices
    g1 = [i for i in ORI_11m1 if 0 <= i < M]
    g2 = [i for i in ORI_m111 if 0 <= i < M]
    n1, n2 = len(g1), len(g2)

    obs, within, cross = block_contrast_abs(C, g1, g2)

    all_idx = np.arange(M)
    stats = np.empty(nperm, float)
    for k in range(nperm):
        perm = rng.permutation(all_idx)
        g1p = perm[:n1]
        g2p = perm[n1:n1+n2]
        stats[k], _, _ = block_contrast_abs(C, g1p, g2p)

    # one-sided p-value: how often perm >= observed
    p = (np.sum(stats >= obs) + 1) / (nperm + 1)
    return {"obs": obs, "within": within, "cross": cross, "p": p, "null": stats}

# Example:import numpy as np

def shuffle_null_eigs(z_mn, nperm=300, seed=1, mode="time_shuffle"):
    """
    Build a null distribution for the top eigenvalue of the NV×NV coherence matrix.

    Parameters
    ----------
    z_mn : (M,N) complex
        Whitened complex series (output out["z"]).
    nperm : int
        Number of permutations.
    seed : int
        RNG seed.
    mode : str
        "time_shuffle"  : independently permute time samples within each NV (destroys cross-NV corr)
        "phase_random"  : randomize phases in Fourier domain per NV, preserve PSD, destroy cross-NV corr

    Returns
    -------
    emax : (nperm,) float
        Null distribution of largest eigenvalue.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z_mn, np.complex128)
    M, N = z.shape

    emax = np.empty(nperm, dtype=float)

    for k in range(nperm):
        if mode == "time_shuffle":
            zs = z.copy()
            for i in range(M):
                rng.shuffle(zs[i])  # in-place along time axis
        elif mode == "phase_random":
            # preserves per-NV power spectrum, removes cross-NV coherence
            zs = np.empty_like(z)
            for i in range(M):
                X = np.fft.rfft(z[i])
                phase = rng.uniform(0, 2*np.pi, size=X.shape)
                Xr = np.abs(X) * np.exp(1j*phase)
                zs[i] = np.fft.irfft(Xr, n=N)
        else:
            raise ValueError("mode must be 'time_shuffle' or 'phase_random'")

        C = coherence_matrix_time(zs)
        w, _ = eig_hermitian(C)
        emax[k] = float(w[0])

    return emax

def correlated_participation_score(C, K=6):
    """
    Score_i = sum_{k=1..K} |v_k(i)|^2 where v_k are eigenvectors of C.
    """
    w, V = eig_hermitian(C)  # V[:,k]
    K = min(K, V.shape[1])
    score = np.sum(np.abs(V[:, :K])**2, axis=1)
    return score, w[:K], V[:, :K]

def plot_nv_map_scores(xy, score, valid=None, top_n=40, title="NV map (score)", invert_y=True):
    xy = np.asarray(xy)
    score = np.asarray(score)
    if valid is None:
        valid = np.all(np.isfinite(xy), axis=1)

    idx_valid = np.where(valid)[0]
    s_valid = score[idx_valid]
    order = idx_valid[np.argsort(s_valid)[::-1]]  # descending score
    top = order[:min(top_n, len(order))]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xy[valid,0], xy[valid,1], s=12, alpha=0.25)
    ax.scatter(xy[top,0], xy[top,1], s=45, alpha=0.9)

    for k, i in enumerate(top[:15]):  # label first ~15 to avoid clutter
        ax.text(xy[i,0], xy[i,1], str(i), fontsize=8)

    ax.set_title(title + f" | top {len(top)} highlighted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, ls="--", lw=0.5)
    ax.axis("equal")
    if invert_y:
        ax.invert_yaxis()  # pixel coords often have y downward
    plt.tight_layout()
    return fig, ax, top

def corr_abs_vs_distance(C, xy, valid=None, nbins=25, subset=None):
    """
    Uses |C_ij| for i<j. Excludes diagonal.
    If subset is provided (list of indices), only uses those indices.
    Returns bin_centers, mean_absC
    """
    C = np.asarray(C)
    xy = np.asarray(xy)
    M = C.shape[0]
    if valid is None:
        valid = np.all(np.isfinite(xy), axis=1)

    idx = np.where(valid)[0]
    if subset is not None:
        subset = np.array([i for i in subset if i in set(idx)], dtype=int)
        idx = subset

    d_list, c_list = [], []
    for a in range(len(idx)):
        i = idx[a]
        for b in range(a+1, len(idx)):
            j = idx[b]
            d = np.linalg.norm(xy[i] - xy[j])
            d_list.append(d)
            c_list.append(np.abs(C[i,j]))

    d_list = np.array(d_list)
    c_list = np.array(c_list)

    bins = np.linspace(d_list.min(), d_list.max(), nbins+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    m = np.full(nbins, np.nan)
    for k in range(nbins):
        sel = (d_list >= bins[k]) & (d_list < bins[k+1])
        if np.any(sel):
            m[k] = np.mean(c_list[sel])
    return centers, m

def plot_corr_vs_r(r, m, title="|C| vs distance"):
    plt.figure(figsize=(6,3.2))
    plt.plot(r, m, "o-")
    plt.xlabel("distance (same units as xy)")
    plt.ylabel("mean |C_ij|")
    plt.title(title)
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


def extract_xy_from_nv_list(nv_list, key="pixel"):
    M = len(nv_list)
    xy = np.full((M,2), np.nan, float)
    for i, nv in enumerate(nv_list):
        c = getattr(nv, "coords", None)
        if c is None: 
            continue
        v = c.get(key, None)
        if v is not None and len(v) >= 2:
            xy[i] = np.array(v[:2], float)
    valid = np.all(np.isfinite(xy), axis=1)
    return xy, valid

def plot_highlight(xy, highlight, title="", invert_y=True, label_first=20):
    xy = np.asarray(xy)
    highlight = np.asarray(highlight, int)

    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.scatter(xy[:,0], xy[:,1], s=10, alpha=0.2)
    ax.scatter(xy[highlight,0], xy[highlight,1], s=15, alpha=0.95)
    for i in highlight[:label_first]:
        ax.text(xy[i,0], xy[i,1], str(i), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.axis("equal"); ax.grid(True, ls="--", lw=0.5)
    if invert_y: ax.invert_yaxis()
    plt.tight_layout()

def cluster_stat_radius(xy, idx):
    """Radius of gyration of selected points."""
    pts = xy[np.asarray(idx, int)]
    ctr = np.mean(pts, axis=0)
    return float(np.sqrt(np.mean(np.sum((pts-ctr)**2, axis=1))))

def perm_test_spatial_cluster(xy, idx, nperm=5000, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, float)
    idx = np.asarray(idx, int)
    n = len(idx)

    obs = cluster_stat_radius(xy, idx)
    all_idx = np.arange(xy.shape[0])
    null = np.empty(nperm, float)
    for k in range(nperm):
        pick = rng.choice(all_idx, size=n, replace=False)
        null[k] = cluster_stat_radius(xy, pick)

    p = (np.sum(null <= obs) + 1) / (nperm + 1)   # smaller radius = more clustered
    return obs, null, p


def close_pairs(xy, idx, rmax=8.0):
    """
    Return list of (i,j,dist) among idx with dist <= rmax.
    rmax in same units as xy (pixels if xy is pixel coords).
    """
    idx = np.asarray(idx, int)
    pts = xy[idx]
    pairs = []
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            d = float(np.linalg.norm(pts[a]-pts[b]))
            if d <= rmax:
                pairs.append((int(idx[a]), int(idx[b]), d))
    pairs.sort(key=lambda x: x[2])
    return pairs


# ---------- helpers ----------
def _nanpct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return np.percentile(x, q)

def mad_zscore(x, eps=1e-12):
    """Robust z-score via median absolute deviation."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * max(mad, eps)
    return (x - med) / scale

def shot_energy_from_z(z_mn):
    """Global per-shot energy; ~1 if perfectly whitened and stable."""
    z = np.asarray(z_mn, np.complex128)
    return np.nanmean(np.abs(z)**2, axis=0)

def find_glitch_shots(energy, zthr=6.0):
    """Return boolean mask for glitch shots using robust z-score."""
    zz = mad_zscore(energy)
    return np.abs(zz) >= zthr, zz


# ---------- core plots ----------
def plot_bad_run_stats(out, global_bad_thresh=0.9):
    frac_run = out.get("frac_global_bad_per_run0", None)
    if frac_run is None:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(frac_run, "o-", ms=3)
    ax.axhline(global_bad_thresh, ls="--")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Frac(global-bad shots)")
    ax.set_title("Bad-run diagnostic")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

def plot_iq_cloud(out, max_points=200000):
    I0 = out.get("I0", None)
    Q0 = out.get("Q0", None)
    if I0 is None or Q0 is None:
        return

    I = np.asarray(I0).ravel()
    Q = np.asarray(Q0).ravel()
    m = np.isfinite(I) & np.isfinite(Q)
    I, Q = I[m], Q[m]
    if I.size == 0:
        return

    if I.size > max_points:
        idx = np.random.choice(I.size, size=max_points, replace=False)
        I, Q = I[idx], Q[idx]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(I, Q, ".", ms=1, alpha=0.25)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("IQ cloud (sanity check)")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    
def plot_iq_cloud_time_colored(out, use="s0", max_points=120000, stride=None,
                               cmap="viridis", alpha=0.35, ms=2):
    """
    Colors IQ points by shot index (time). If it traces a path -> drift.
    use: 's0' (recommended) or 'z' (whitened)
    """
    s = out.get(use, None)
    if s is None:
        raise KeyError(f"out['{use}'] not found")

    s = np.asarray(s)
    # s shape: (M,N)
    I = np.real(s).ravel()
    Q = np.imag(s).ravel()
    t = np.tile(np.arange(s.shape[1]), s.shape[0])  # repeat shot index for each NV

    m = np.isfinite(I) & np.isfinite(Q)
    I, Q, t = I[m], Q[m], t[m]

    if I.size == 0:
        print("No finite IQ points.")
        return

    # Optional deterministic stride (faster + preserves time ordering)
    if stride is not None and stride > 1:
        sel = np.arange(I.size) % int(stride) == 0
        I, Q, t = I[sel], Q[sel], t[sel]

    # Random downsample (keeps speed)
    if I.size > max_points:
        idx = np.random.choice(I.size, size=max_points, replace=False)
        I, Q, t = I[idx], Q[idx], t[idx]

    # Normalize t to [0,1] for colormap
    tn = (t - t.min()) / max(1, (t.max() - t.min()))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    sc = ax.scatter(I, Q, c=tn, s=ms, alpha=alpha, cmap=cmap, linewidths=0)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"IQ cloud colored by time (use='{use}')")
    ax.grid(True, ls="--", lw=0.5)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("normalized shot index")
    plt.tight_layout()

def plot_global_vs_residual_cloud(out, use="s0", max_points=150000,
                                 alpha=0.30, ms=2):
    """
    Left: global mean g[k] cloud (one point per shot)
    Right: residuals r[n,k] = s[n,k] - g[k] cloud (many points)

    If only the global mean has structure -> global common-mode.
    """
    s = out.get(use, None)
    if s is None:
        raise KeyError(f"out['{use}'] not found")
    s = np.asarray(s)  # (M,N)

    # Global mean per shot
    g = np.nanmean(s, axis=0)  # (N,)

    # Residuals
    r = s - g[None, :]

    # Prepare points
    Ig, Qg = np.real(g), np.imag(g)
    Ir, Qr = np.real(r).ravel(), np.imag(r).ravel()

    mg = np.isfinite(Ig) & np.isfinite(Qg)
    mr = np.isfinite(Ir) & np.isfinite(Qr)

    Ig, Qg = Ig[mg], Qg[mg]
    Ir, Qr = Ir[mr], Qr[mr]

    # Downsample residuals for speed
    if Ir.size > max_points:
        idx = np.random.choice(Ir.size, size=max_points, replace=False)
        Ir, Qr = Ir[idx], Qr[idx]

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.8))

    # Global mean cloud
    ax[0].scatter(Ig, Qg, s=12, alpha=0.7, linewidths=0)
    ax[0].set_title("Global mean cloud: g[k] = mean_n s[n,k]")
    ax[0].set_xlabel("I")
    ax[0].set_ylabel("Q")
    ax[0].grid(True, ls="--", lw=0.5)

    # Residual cloud
    ax[1].scatter(Ir, Qr, s=ms, alpha=alpha, linewidths=0)
    ax[1].set_title("Residual cloud: r[n,k] = s[n,k] - g[k]")
    ax[1].set_xlabel("I")
    ax[1].set_ylabel("Q")
    ax[1].grid(True, ls="--", lw=0.5)

    # Match axes to compare shapes fairly
    allI = np.r_[Ig, Ir]
    allQ = np.r_[Qg, Qr]
    if allI.size and allQ.size:
        xlo, xhi = np.percentile(allI[np.isfinite(allI)], [1, 99])
        ylo, yhi = np.percentile(allQ[np.isfinite(allQ)], [1, 99])
        for a in ax:
            a.set_xlim(xlo, xhi)
            a.set_ylim(ylo, yhi)

    plt.tight_layout()


def plot_example_traces(out, nv_indices=(0, 1, 2, 3), nshots=800):
    z = out.get("z", None)
    if z is None:
        return
    z = np.asarray(z)
    M, N = z.shape
    nshots = min(int(nshots), N)
    t = np.arange(nshots)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i in nv_indices:
        if 0 <= int(i) < M:
            ax.plot(t, np.real(z[int(i), :nshots]), lw=1, label=f"NV {i}")
    ax.set_xlabel("shot index")
    ax.set_ylabel("Re(z) (whitened)")
    ax.set_title("Example whitened traces")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

def plot_matrix_abs(C, title="|C|", prc=(1, 99), diag_color="0.85"):
    C = np.asarray(C, np.complex128)
    A = np.abs(C)

    # autoscale off-diagonal
    A_tmp = A.copy()
    np.fill_diagonal(A_tmp, np.nan)
    vmin, vmax = np.nanpercentile(A_tmp, prc)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(A_tmp), np.nanmax(A_tmp)

    mask = np.eye(A.shape[0], dtype=bool)
    Am = np.ma.array(A, mask=mask)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(diag_color)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("NV index")
    ax.set_ylabel("NV index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


# def plot_common_mode_and_glitches(out, dt=1.0, zthr=10.0, nshots=None):
#     a = out.get("a_time", None)
#     z = out.get("z", None)
#     if a is None or z is None:
#         return

#     a = np.asarray(a)
#     z = np.asarray(z)
#     E = shot_energy_from_z(z)
#     glitch, Ez = find_glitch_shots(E, zthr=zthr)

#     if nshots is not None:
#         nshots = min(int(nshots), a.size)
#         a = a[:nshots]
#         E = E[:nshots]
#         glitch = glitch[:nshots]
#         Ez = Ez[:nshots]

#     t = np.arange(a.size) * float(dt)

#     # time series
#     fig, ax = plt.subplots(figsize=(9, 3.2))
#     ax.plot(t, np.real(a), lw=1, label="Re(a)")
#     ax.plot(t, np.imag(a), lw=1, label="Im(a)")
#     if np.any(glitch):
#         ax.plot(t[glitch], np.real(a)[glitch], "rx", ms=5, label="glitch shots")
#     ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
#     ax.set_ylabel("a(t)")
#     ax.set_title("Top correlated mode a(t) (with glitch markers)")
#     ax.grid(True, ls="--", lw=0.5)
#     ax.legend(fontsize=8, ncol=3)
#     plt.tight_layout()

#     # shot energy
#     fig, ax = plt.subplots(figsize=(9, 4.0))
#     ax.plot(t, E, lw=1)
#     ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
#     ax.set_ylabel("mean |z|^2 across NVs")
#     ax.set_title(f"Shot energy (robust z-score threshold = {zthr})")
#     ax.grid(True, ls="--", lw=0.5)
#     # show robust scale
#     ax2 = ax.twinx()
#     ax2.plot(t, Ez, alpha=0.0)  # just to set scale if needed
#     plt.tight_layout()

#     print("shot_energy percentiles:", np.nanpercentile(E, [50, 90, 95, 99, 99.5, 99.9]))
#     print("glitch shots fraction:", float(np.mean(glitch)))


def shot_energy_from_z(z):
    """E[k] = mean_n |z[n,k]|^2"""
    z = np.asarray(z)
    return np.mean(np.abs(z)**2, axis=0)

def robust_zscore_mad(x, eps=1e-12):
    """Robust z-score using median and MAD."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad + eps
    return (x - med) / sigma, med, sigma

def find_glitch_shots(E, zthr=10.0):
    """Return glitch mask where robust z-score(E) > zthr."""
    Ez, med, sig = robust_zscore_mad(E)
    glitch = Ez > float(zthr)
    return glitch, Ez, med, sig

def plot_common_mode_and_glitches(out, dt=1.0, zthr=10.0, nshots=None,
                                  lag=1, amp_min_percentile=10.0, fmax=None):
    """
    Plots:
      (1) common-mode a(t): Re/Im + amplitude
      (2) common-mode phase (unwrapped) with masking for low amplitude
      (3) phase increments dphi(t; lag)
      (4) shot energy E[k] + robust z-score + glitch markers
    """
    a = out.get("a_time", None)
    z = out.get("z", None)
    if a is None or z is None:
        raise ValueError("out must contain 'a_time' and 'z'")

    a = np.asarray(a, np.complex128)
    z = np.asarray(z, np.complex128)

    E = shot_energy_from_z(z)
    glitch, Ez, Em, Es = find_glitch_shots(E, zthr=zthr)

    # optional truncation
    N = a.size
    if nshots is not None:
        N = min(int(nshots), N)
        a = a[:N]
        E = E[:N]
        glitch = glitch[:N]
        Ez = Ez[:N]

    t = np.arange(N) * float(dt)

    # common-mode amplitude/phase
    amp = np.abs(a)
    phi = np.unwrap(np.angle(a))

    # mask phase when amplitude is too small
    thr = np.nanpercentile(amp, float(amp_min_percentile))
    phase_mask = amp >= thr
    phi_masked = np.where(phase_mask, phi, np.nan)

    # phase increments
    if lag >= 1 and N > lag:
        dphi = np.angle(a[lag:] * np.conj(a[:-lag]))  # in (-pi, pi]
        td = t[:-lag]  # aligns with dphi
    else:
        dphi = None
        td = None

    # --- (1) a(t): Re/Im + glitch markers ---
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, np.real(a), lw=1, label="Re(a)")
    ax.plot(t, np.imag(a), lw=1, label="Im(a)")
    if np.any(glitch):
        ax.plot(t[glitch], np.real(a)[glitch], "rx", ms=5, label="glitch shots")
    ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
    ax.set_ylabel("a(t)")
    ax.set_title("Top correlated mode a(t)")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()

    # --- (1b) |a(t)| ---
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t, amp, lw=1)
    if np.any(glitch):
        ax.plot(t[glitch], amp[glitch], "rx", ms=5)
    ax.axhline(thr, ls="--", lw=1)
    ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
    ax.set_ylabel("|a(t)|")
    ax.set_title(f"Common-mode amplitude |a(t)| (phase mask threshold = p{amp_min_percentile:g})")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

    # --- (2) phase ---
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(t, phi_masked, lw=1)
    if np.any(glitch):
        ax.plot(t[glitch], phi_masked[glitch], "rx", ms=5)
    ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
    ax.set_ylabel("unwrap(angle(a)) [rad]")
    ax.set_title("Common-mode phase (masked when |a| is small)")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()

    # --- (3) phase increments ---
    if dphi is not None:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.plot(td, dphi, lw=1)
        ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
        ax.set_ylabel(f"dphi(t, lag={lag}) [rad]")
        ax.set_title("Common-mode phase increments (shot-to-shot phase noise proxy)")
        ax.grid(True, ls="--", lw=0.5)
        plt.tight_layout()

    # --- (4) shot energy ---
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(t, E, lw=1, label="E[k]=mean |z|^2")
    if np.any(glitch):
        ax.plot(t[glitch], E[glitch], "rx", ms=5, label="glitch shots")
    ax.set_xlabel("time (s)" if dt != 1.0 else "shot index")
    ax.set_ylabel("shot energy E[k]")
    ax.set_title(f"Shot energy (robust z-score > {zthr}; median={Em:.4g}, robustσ={Es:.4g})")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()

    # --- optional: spectrum of a(t) ---
    # (useful if you want to see lines in the common mode)
    A = np.fft.rfft((a[:N] - np.mean(a[:N])) * np.hanning(N))
    f = np.fft.rfftfreq(N, d=float(dt))
    psd = np.abs(A)**2
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(f, psd, lw=1)
    ax.set_xlabel("Hz" if dt != 1.0 else "cycles/shot")
    ax.set_ylabel("|A(f)|^2")
    ax.set_title("Spectrum of a(t)")
    ax.grid(True, ls="--", lw=0.5)
    if fmax is not None:
        ax.set_xlim(0, fmax)
    plt.tight_layout()

    print("shot_energy percentiles:", np.nanpercentile(E, [50, 90, 95, 99, 99.5, 99.9]))
    print("glitch shots fraction:", float(np.mean(glitch)))


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Core helpers
# ----------------------------
def shot_energy(x_mn):
    """Per-shot mean power across NVs. x_mn: (M,N) complex."""
    x = np.asarray(x_mn)
    return np.nanmean(np.abs(x)**2, axis=0)

def global_mean_series(x_mn):
    """g[k] = mean_n x[n,k]."""
    x = np.asarray(x_mn)
    return np.nanmean(x, axis=0)

def split_common_and_residual(x_mn):
    """
    x_mn: (M,N)
    g: (N,) global mean series
    r: (M,N) residuals after removing g[k]
    """
    x = np.asarray(x_mn)
    g = global_mean_series(x)          # (N,)
    r = x - g[None, :]                 # (M,N)
    return g, r

def robust_zscore(x_1d, eps=1e-12):
    """Robust z-score using median/MAD."""
    x = np.asarray(x_1d, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad
    sigma = max(float(sigma), eps)
    return (x - med) / sigma

def find_glitch_shots(E, zthr=8.0):
    """
    E: (N,) shot energy series
    Returns glitch_mask (N,), Ez (N,) robust z-score
    """
    Ez = robust_zscore(E)
    glitch = np.abs(Ez) >= float(zthr)
    return glitch, Ez

# ----------------------------
# Main plotter
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def robust_zscore_mad(x, eps=1e-12):
    """Robust z-score using median + MAD."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sigma = 1.4826 * mad + eps
    z = (x - med) / sigma
    return z, med, sigma

def shot_energy(x_mn):
    """E[k] = mean_n |x[n,k]|^2. x_mn: (M,N) complex."""
    x = np.asarray(x_mn)
    return np.nanmean(np.abs(x)**2, axis=0)

def split_common_and_residual(x_mn):
    """g[k] = mean_n x[n,k], r[n,k] = x[n,k] - g[k]."""
    x = np.asarray(x_mn)
    g = np.nanmean(x, axis=0)          # (N,)
    r = x - g[None, :]                 # (M,N)
    return g, r

def plot_physical_shot_diagnostics_s0(
    out,
    nshots=None,
    dt=1.0,
    zthr=8.0,
    amp_prc=20.0,
    mark_glitches=True,
):
    """
    Physical-only diagnostics from out["s0"] (complex lock-in series).

    Plots:
      1) Shot energy E[k] = <|s0|^2>_n  (+ glitch markers via robust z-score)
      2) Residual energy after removing global mean g[k] = <s0>_n
      3) |g[k]| (global mean amplitude)
      4) dphi[k] = angle(g[k] * conj(g[k-1])) (masked when |g| is small)
    """
    if "s0" not in out:
        raise KeyError("out does not contain 's0'. This function is physical-only and expects out['s0'].")

    x = np.asarray(out["s0"])
    if x.ndim != 2:
        raise ValueError(f"Expected out['s0'] shape (M,N), got {x.shape}")

    if nshots is not None:
        nshots = min(int(nshots), x.shape[1])
        x = x[:, :nshots]

    M, N = x.shape
    t = np.arange(N, dtype=float) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    # Common-mode split
    g, r = split_common_and_residual(x)

    # Energies
    E = shot_energy(x)
    Eres = shot_energy(r)

    # Glitches from E
    zE, Em, Es = robust_zscore_mad(E)
    glitch = np.abs(zE) >= float(zthr)

    # Global amplitude + phase increment
    amp = np.abs(g)
    amp_thr = np.nanpercentile(amp, float(amp_prc))

    # phase increment is the physical-ish thing (absolute phase is arbitrary)
    dphi = np.angle(g[1:] * np.conj(g[:-1]))  # (N-1,)

    # Mask dphi where global mean is too small on either shot
    good = amp > amp_thr
    good_d = good[1:] & good[:-1]
    dphi_m = np.where(good_d, dphi, np.nan)

    # # ----- PLOTS -----
    # fig, ax = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True)

    # # 1) Shot energy
    # ax[0].plot(t, E, lw=1)
    # if mark_glitches and np.any(glitch):
    #     ax[0].plot(t[glitch], E[glitch], "rx", ms=4, label=f"glitch |zE|≥{zthr:g}")
    #     ax[0].legend(fontsize=8, loc="upper right")
    # ax[0].set_ylabel(r"$\langle |s_0|^2\rangle_n$")
    # ax[0].set_title("Shot energy (global amplitude events)")

    # # 2) Residual energy
    # ax[1].plot(t, Eres, lw=1)
    # if mark_glitches and np.any(glitch):
    #     ax[1].plot(t[glitch], Eres[glitch], "rx", ms=4)
    # ax[1].set_ylabel(r"$\langle |s_0-g|^2\rangle_n$")
    # ax[1].set_title("Residual energy after removing global mean (tests ‘pure common-mode’)")

    # # 3) |g|
    # ax[2].plot(t, amp, lw=1, label=r"$|g[k]|$")
    # ax[2].axhline(amp_thr, ls="--", lw=1, label=f"|g| threshold (p{amp_prc:g})")
    # if mark_glitches and np.any(glitch):
    #     ax[2].plot(t[glitch], amp[glitch], "rx", ms=4)
    # ax[2].set_ylabel(r"$|g[k]|$")
    # ax[2].set_xlabel(xlab)
    # ax[2].set_title("Global mean amplitude (common-mode strength)")
    # ax[2].legend(fontsize=8, loc="upper right")

    # for a in ax:
    #     a.grid(True, ls="--", lw=0.5)
    # plt.tight_layout()
    # plt.show()

    # # ----- PLOTS -----
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.0))
    # 1) Shot energy
    ax.plot(t, E, lw=1)
    if mark_glitches and np.any(glitch):
        ax.plot(t[glitch], E[glitch], "rx", ms=4, label=f"glitch |zE|≥{zthr:g}")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel(r"$\langle |s_0|^2\rangle_n$")
    ax.set_xlabel(xlab)
    ax.set_title("Shot energy (global amplitude events)")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    
    # 4) Phase increments (separate, clearer scaling)
    # fig, axp = plt.subplots(1, 1, figsize=(10, 3.0))
    # axp.plot(t[1:], dphi_m, lw=1)
    # axp.set_xlabel(xlab)
    # axp.set_ylabel(r"$\Delta\phi[k]$ (rad)")
    # axp.set_title(r"Global phase increment $\Delta\phi[k]=\arg(g[k]g^*[k-1])$ (masked when |g| small)")
    # axp.grid(True, ls="--", lw=0.5)
    # plt.tight_layout()
    # plt.show()

    # ----- PRINTS -----
    prc = [1, 5, 50, 95, 99]
    print("E percentiles:", np.nanpercentile(E, prc))
    print("Eres percentiles:", np.nanpercentile(Eres, prc))
    print("amp_thr:", float(amp_thr), f"(mask keeps {100*np.nanmean(good):.1f}% shots)")
    print("glitch fraction:", float(np.nanmean(glitch)))
    print("zE percentiles:", np.nanpercentile(zE, [1, 50, 99]))

    return {
        "t": t,
        "E": E,
        "Eres": Eres,
        "zE": zE,
        "glitch": glitch,
        "g": g,
        "amp": amp,
        "amp_thr": amp_thr,
        "dphi": dphi,
        "dphi_masked": dphi_m,
    }

import numpy as np
import matplotlib.pyplot as plt

def _phase_inc(z):
    """Δphi[k] = angle(z[k] * conj(z[k-1])) for 1D complex series."""
    return np.angle(z[1:] * np.conj(z[:-1]))

def _unwrap_phase(z):
    """Unwrapped phase of 1D complex series."""
    return np.unwrap(np.angle(z))

def _safe_mean(x, axis=0):
    return np.nanmean(x, axis=axis)

def _safe_amp(z):
    return np.abs(z)

def sample_from_groups(groups, n_each=6, seed=0):
    """Return dict name -> sampled indices (no replacement)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, idx in groups.items():
        idx = np.array(idx, dtype=int)
        if len(idx) == 0:
            out[name] = np.array([], dtype=int)
            continue
        k = min(n_each, len(idx))
        out[name] = rng.choice(idx, size=k, replace=False)
    return out

def plot_orientation_commonmode(
    out,
    groups,
    dt=1.0,
    nshots=None,
    amp_prc=20.0,
    windows_s=None,   # list of (t0,t1) in seconds
    title_prefix="",
):
    """
    Plots common-mode phasor for:
      - all NVs
      - each orientation group (mean over NVs in group)

    Shows:
      (1) |g_all(t)| and |g_group(t)|
      (2) Δphi_all(t) and Δphi_group(t)   (masked when amplitude low)

    Interpretation:
      - If Δphi jumps are similar across groups -> reference/timing/MW phase.
      - If Δphi jump sizes differ by group -> real ΔB projection effect.
      - If mostly |g| changes -> optical/contrast/global amplitude change.
    """
    s = np.asarray(out["s0"])  # (M,N)
    if nshots is not None:
        nshots = min(int(nshots), s.shape[1])
        s = s[:, :nshots]
    M, N = s.shape
    t = np.arange(N) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    # Global mean phasor across all NVs
    g_all = _safe_mean(s, axis=0)             # (N,)
    amp_all = _safe_amp(g_all)
    amp_thr_all = np.nanpercentile(amp_all, amp_prc)
    good_all = amp_all > amp_thr_all
    dphi_all = _phase_inc(g_all)
    dphi_all_m = np.where(good_all[1:] & good_all[:-1], dphi_all, np.nan)

    # Per-group phasors
    gG = {}
    ampG = {}
    dphiG_m = {}
    amp_thrG = {}
    for name, idx in groups.items():
        idx = np.array(idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < M)]
        if len(idx) == 0:
            continue
        g = _safe_mean(s[idx, :], axis=0)
        a = _safe_amp(g)
        thr = np.nanpercentile(a, amp_prc)
        good = a > thr
        dphi = _phase_inc(g)
        dphi_m = np.where(good[1:] & good[:-1], dphi, np.nan)
        gG[name] = g
        ampG[name] = a
        dphiG_m[name] = dphi_m
        amp_thrG[name] = thr

    # ---- Plot amplitudes ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.2))
    ax.plot(t, amp_all, lw=1.2, label="|g_all|")
    ax.axhline(amp_thr_all, ls="--", lw=1, label=f"|g_all| p{amp_prc:g}")
    for name in gG:
        ax.plot(t, ampG[name], lw=1, alpha=0.9, label=f"|g_{name}|")
    if windows_s:
        for (t0, t1) in windows_s:
            ax.axvspan(t0, t1, alpha=0.15)
    ax.set_title(f"{title_prefix} Common-mode amplitudes")
    ax.set_xlabel(xlab); ax.set_ylabel("|g(t)|")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    plt.tight_layout()
    plt.show()

    # ---- Plot phase increments ----
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.2))
    ax.plot(t[1:], dphi_all_m, lw=1.2, label="Δphi_all")
    for name in dphiG_m:
        ax.plot(t[1:], dphiG_m[name], lw=1, alpha=0.9, label=f"Δphi_{name}")
    if windows_s:
        for (t0, t1) in windows_s:
            ax.axvspan(t0, t1, alpha=0.15)
    ax.set_title(f"{title_prefix} Common-mode phase increments (masked when |g| small)")
    ax.set_xlabel(xlab); ax.set_ylabel("Δphi (rad)")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    plt.tight_layout()
    plt.show()

    return {
        "t": t,
        "g_all": g_all,
        "amp_all": amp_all,
        "dphi_all_masked": dphi_all_m,
        "g_groups": gG,
        "amp_groups": ampG,
        "dphi_groups_masked": dphiG_m,
    }

def plot_random_nv_traces(
    out,
    nv_indices,
    dt=1.0,
    nshots=None,
    windows_s=None,
    show="phase",     # "phase" or "amp" or "IQ"
    unwrap=True,
    title="Random NV traces",
):
    """
    Plot a handful of NV traces.
    - show="phase": plots angle(s0) (unwrapped optionally)
    - show="amp": plots |s0|
    - show="IQ": plots I and Q (two panels)
    """
    s = np.asarray(out["s0"])
    if nshots is not None:
        nshots = min(int(nshots), s.shape[1])
        s = s[:, :nshots]
    M, N = s.shape
    t = np.arange(N) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    idx = np.array(nv_indices, dtype=int)
    idx = idx[(idx >= 0) & (idx < M)]
    if len(idx) == 0:
        print("No valid NV indices.")
        return

    if show.lower() == "iq":
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        for i in idx:
            ax[0].plot(t, np.real(s[i]), lw=1, alpha=0.9, label=f"NV {i}")
            ax[1].plot(t, np.imag(s[i]), lw=1, alpha=0.9, label=f"NV {i}")
        if windows_s:
            for (t0, t1) in windows_s:
                ax[0].axvspan(t0, t1, alpha=0.15)
                ax[1].axvspan(t0, t1, alpha=0.15)
        ax[0].set_ylabel("I"); ax[1].set_ylabel("Q")
        ax[1].set_xlabel(xlab)
        ax[0].set_title(title + " (I)"); ax[1].set_title(title + " (Q)")
        for a in ax: a.grid(True, ls="--", lw=0.5)
        plt.tight_layout(); plt.show()
        return

    fig, ax = plt.subplots(1, 1, figsize=(11, 3.2))
    for i in idx:
        if show.lower() == "amp":
            y = np.abs(s[i])
        else:
            y = np.angle(s[i])
            if unwrap:
                y = np.unwrap(y)
        ax.plot(t, y, lw=1, alpha=0.9, label=f"NV {i}")
    if windows_s:
        for (t0, t1) in windows_s:
            ax.axvspan(t0, t1, alpha=0.15)
    ax.set_xlabel(xlab)
    ax.set_ylabel("|s0|" if show.lower()=="amp" else "phase (rad)")
    ax.set_title(title + f" ({show})")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    plt.tight_layout()
    plt.show()

def remove_common_phase(s_mn, eps=1e-12):
    """
    Removes ONLY a global phase rotation:
      s_i(t) -> s_i(t) * exp(-j*phi_cm(t))
    where phi_cm(t) is the phase of the global mean g_all(t).
    Useful test:
      - if correlated jumps vanish after this, it was likely a reference-phase/timing jump.
    """
    s = np.asarray(s_mn)
    g = np.nanmean(s, axis=0)
    ph = np.angle(g)
    rot = np.exp(-1j * ph)
    # avoid rotating by garbage when |g| ~ 0
    mask = (np.abs(g) > eps)
    rot2 = np.ones_like(rot)
    rot2[mask] = rot[mask]
    return s * rot2[None, :]



def make_summary_plots(out, dt=1.0, global_bad_thresh=0.9, zthr=6.0):
    # plot_bad_run_stats(out, global_bad_thresh=global_bad_thresh)
    plot_iq_cloud(out)
    plot_iq_cloud_time_colored(out, use="s0")
    plot_global_vs_residual_cloud(out, use="s0")

    plot_example_traces(out, nv_indices=(0, 1, 2, 3), nshots=8000)

    if "C_time" in out:
        plot_matrix_abs(out["C_time"], title="Time-domain |C| (NV×NV)")
    if "evals_time" in out:
        plot_eigs(out["evals_time"], title="Eigenvalues of time-domain C")

    plot_common_mode_and_glitches(out, dt=dt, zthr=zthr)
    plt.show()


def top_by_mass(v, frac=0.8):
    """
    Smallest set of indices whose sum(|v|^2) reaches `frac`.
    v: complex eigenvector (length M)
    """
    v = np.asarray(v)
    w = np.abs(v)**2
    w = w / np.sum(w)
    order = np.argsort(w)[::-1]
    c = np.cumsum(w[order])
    k = int(np.searchsorted(c, frac) + 1)
    return order[:k], w, order, c

# ----------------------------
# Helpers (keep small + reusable)
# ----------------------------
def shot_energy(z_mn):
    """Per-shot mean power across NVs. z_mn: (M,N) complex."""
    z = np.asarray(z_mn, np.complex128)
    return np.mean(np.abs(z) ** 2, axis=0)

def per_shot_normalize(z_mn, eps=1e-12):
    """Normalize each shot so mean power across NVs is 1."""
    z = np.asarray(z_mn, np.complex128)
    E = shot_energy(z)
    return z / np.sqrt(np.maximum(E, eps))[None, :], E

def top_eig_from_z(z_mn):
    """Return (lambda1, evals, v1) from coherence_matrix_time(z_mn)."""
    C = coherence_matrix_time(z_mn)
    w, V = eig_hermitian(C)
    return float(w[0]), w, V[:, 0]

def remove_mode(z_mn, v_m):
    """Project out rank-1 mode: z -> z - v (v^H z)."""
    z = np.asarray(z_mn, np.complex128)
    v = np.asarray(v_m, np.complex128).reshape(-1)
    a = v.conj().T @ z              # (N,)
    return z - np.outer(v, a)       # (M,N)

def top_by_mass(v_m, frac=0.80):
    """Smallest index set whose cumulative |v|^2 mass reaches frac."""
    v = np.asarray(v_m, np.complex128).reshape(-1)
    p = np.abs(v) ** 2
    p = p / np.sum(p)
    order = np.argsort(p)[::-1]
    cdf = np.cumsum(p[order])
    k = int(np.searchsorted(cdf, frac) + 1)
    return order[:k], p, order, cdf

def participation_ratio(v_m, eps=1e-30):
    """PR = 1/sum p_i^2 with p_i=|v_i|^2 normalized."""
    v = np.asarray(v_m, np.complex128).reshape(-1)
    p = np.abs(v) ** 2
    p = p / np.maximum(np.sum(p), eps)
    return float(1.0 / np.sum(p ** 2))

def fraction_in_group(indices, group):
    s = set(group)
    idx = np.asarray(indices, int)
    return float(np.mean([i in s for i in idx]))

# ----------------------------
# Main “sanity + interpretation” block
# ----------------------------
def analyze_correlated_modes(
    out,
    nv_list,
    ORI_11m1=None,
    ORI_m111=None,
    K=6,
    prc_clip=(99.0, 99.5, 99.9),
    mass_frac=0.80,
    nbins_all=25,
    nbins_top=15,
    top_subset=40,
):
    z = out["z"]            # (M,N)
    C = out["C_time"]       # (M,M)

    # 1) Shot-energy sanity + per-shot normalization effect
    lam1_raw, w_raw, v1_raw = top_eig_from_z(z)

    zN, E = per_shot_normalize(z)
    lam1_norm, w_norm, v1_norm = top_eig_from_z(zN)

    print("Top evals (raw)      :", w_raw[:K])
    print("Top evals (shot-norm) :", w_norm[:K])

    pcts = [0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    vals = np.percentile(E, pcts)
    print("shot_energy percentiles:")
    for q, v in zip(pcts, vals):
        print(f"{q:>5}%  {v:.4f}")
    print("min / max:", float(np.min(E)), float(np.max(E)))

    # 2) Sensitivity to clipping high-energy shots
    for prc in prc_clip:
        thr = np.percentile(E, prc)
        keep = E <= thr
        lam1_clip, _, _ = top_eig_from_z(z[:, keep])
        print(f"drop top {(100-prc):.1f}% energy shots -> keep {keep.mean()*100:.2f}% shots, lam1={lam1_clip:.6f}")

    # 3) Remove the top mode and recompute lambda1
    z_perp = remove_mode(z, v1_raw)
    lam1_after, w_after, v1_after = top_eig_from_z(z_perp)
    print("lam1 before:", lam1_raw)
    print("lam1 after removing mode1:", lam1_after)

    # 4) Eigenmodes / participation maps
    w_eval, V = eig_hermitian(C)
    score = np.sum(np.abs(V[:, :K]) ** 2, axis=1)  # participation across top-K modes

    print("Top evals:", w_eval[:K])
    idx_top = np.argsort(score)[::-1][:20]
    print("Top 20 NVs by participation score:", idx_top.tolist())
    print("Scores:", score[idx_top])

    plt.figure(figsize=(7,3))
    plt.plot(score, "o", ms=3)
    plt.xlabel("NV index"); 
    # plt.ylabel(r"$\sum_{k \le K} |v_k|^2$")
    plt.ylabel("sum_{k<=K} |v_k|^2")
    plt.title("NV participation in top correlated modes")
    plt.grid(True, ls="--", lw=0.5); plt.tight_layout(); plt.show()

    # matrix sorted by score
    order = np.argsort(score)[::-1]
    Cp = C[np.ix_(order, order)]
    plot_matrix(Cp, title="C_time sorted by participation score", show_phase=False)

    # XY maps
    xy_pix, valid = extract_xy_from_nv_list(nv_list, key="pixel")
    fig, ax, top = plot_nv_map_scores(
        xy_pix, score, valid=valid, top_n=min(200, len(score)),
        title="Pixel map: participation score"
    )
    print("Top indices:", top[:20])

    # distance dependence
    r_all, m_all = corr_abs_vs_distance(C, xy_pix, valid=valid, nbins=nbins_all)
    plot_corr_vs_r(r_all, m_all, title="C_time: mean |C_ij| vs pixel distance (all NVs)")

    topN = np.argsort(score)[::-1][:top_subset]
    r_top, m_top = corr_abs_vs_distance(C, xy_pix, valid=valid, nbins=nbins_top, subset=topN)
    plot_corr_vs_r(r_top, m_top, title=f"C_time: mean |C_ij| vs distance (top-{top_subset})")

    # orientation fractions (optional)
    if ORI_11m1 is not None and ORI_m111 is not None:
        print("Top subset fraction ORI_11m1:", fraction_in_group(topN, ORI_11m1))
        print("Top subset fraction ORI_m111:", fraction_in_group(topN, ORI_m111))

    # 5) Per-mode “who contributes” via mass cutoff (clean visualization set)
    for k in range(K):
        vk = V[:, k]
        pr = participation_ratio(vk)
        idx80, _, _, _ = top_by_mass(vk, frac=mass_frac)
        idx95, _, _, _ = top_by_mass(vk, frac=0.95)

        print(f"mode {k+1}: eval={w_eval[k]:.6f}, PR~{pr:.1f}, "
              f"N{int(mass_frac*100)}={len(idx80)}, N95={len(idx95)}")

        plot_highlight(
            xy_pix, idx80,
            title=f"Mode {k+1}: top contributors ({int(mass_frac*100)}% mass)",
            invert_y=True, label_first=25
        )
        plt.show()

    # 6) Brightness vs participation (optional diagnostic)
    bright = np.mean(np.abs(out["s0"]), axis=1)  # proxy
    plt.figure(figsize=(4,4))
    plt.plot(bright, score, "o", ms=3)
    plt.xlabel("mean |s0| (brightness proxy)")
    # plt.ylabel(r"$\sum_{k\le K}|v_k|^2$")
    plt.ylabel("sum_{k<=K} |v_k|^2")
    plt.title("Brightness vs correlated participation")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout(); plt.show()
    print("corrcoef:", float(np.corrcoef(bright, score)[0, 1]))

    return {
        "lam1_raw": lam1_raw,
        "lam1_shotnorm": lam1_norm,
        "lam1_after_remove_mode1": lam1_after,
        "E_shot": E,
        "w": w_eval,
        "V": V,
        "score": score,
        "xy_pix": xy_pix,
        "valid": valid,
    }

import numpy as np
import matplotlib.pyplot as plt

def _offdiag_vlims(A, prc=(1, 99)):
    """Robust color limits using off-diagonal percentiles."""
    A = np.array(A, float, copy=True)
    np.fill_diagonal(A, np.nan)
    vmin, vmax = np.nanpercentile(A, prc)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(A), np.nanmax(A)
    return vmin, vmax

def _imshow_diag_gray(ax, A, vmin=None, vmax=None, cmap=None, diag_gray="0.85"):
    """imshow with diagonal masked to gray."""
    A = np.asarray(A)
    mask = np.eye(A.shape[0], dtype=bool)
    Am = np.ma.array(A, mask=mask)

    if cmap is None:
        cmap = plt.cm.viridis.copy()
    else:
        cmap = plt.get_cmap(cmap).copy()

    cmap.set_bad(diag_gray)
    im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    return im

def plot_complex_matrix(G, title="Complex matrix", show_phase=True, prc_abs=(1, 99),
                        mask_phase_below=0.0):
    """
    G: (M,M) complex (e.g., coherence matrix)
    mask_phase_below: if >0, phase is masked where |G| < threshold (recommended ~0.05-0.1)
    """
    G = np.asarray(G, np.complex128)

    # --- |G| ---
    A = np.abs(G)
    vmin, vmax = _offdiag_vlims(A, prc=prc_abs)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = _imshow_diag_gray(ax, A, vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(f"{title} |G|")
    ax.set_xlabel("NV index")
    ax.set_ylabel("NV index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    # --- phase(G) ---
    if show_phase:
        P = np.angle(G)

        if mask_phase_below and mask_phase_below > 0:
            P = np.array(P, float, copy=True)
            P[np.abs(G) < float(mask_phase_below)] = np.nan

        # phase scale fixed to [-pi, pi]
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = _imshow_diag_gray(ax, P, vmin=-np.pi, vmax=np.pi, cmap="twilight")
        ax.set_title(f"{title} phase(G)  (masked where |G|<{mask_phase_below:g})" if mask_phase_below else f"{title} phase(G)")
        ax.set_xlabel("NV index")
        ax.set_ylabel("NV index")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
        plt.tight_layout()

def plot_real_matrix(R, title="Real matrix", prc=(1, 99), symmetric=True):
    """
    R: (M,M) real (e.g., correlation matrix)
    symmetric: if True, use symmetric color limits about 0.
    """
    R = np.asarray(R, float)
    if symmetric:
        vmax = np.nanpercentile(np.abs(R - np.diag(np.diag(R))), prc[1])  # robust
        vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else float(np.nanmax(np.abs(R)))
        vmin = -vmax
    else:
        vmin, vmax = _offdiag_vlims(R, prc=prc)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = _imshow_diag_gray(ax, R, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xlabel("NV index")
    ax.set_ylabel("NV index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

def plot_top_eigs(C, K=20, title="Top eigenvalues"):
    C = np.asarray(C)
    # Hermitian eigs if complex coherence; for real corr it's also fine
    Csym = 0.5 * (C + C.conj().T)
    w = np.linalg.eigvalsh(Csym)
    w = np.sort(np.real(w))[::-1]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(w[:K], "o-")
    ax.set_xlabel("mode index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    return w

# ---------- Example “main” plotting wrapper ----------
def plot_all_pairwise(out, use="z", mask_phase_below=0.):
    """
    out: your analysis dict
    use: "z" for whitened complex series, or "s0" for pre-whitened complex series
    """
    x = out[use]  # (M,N) complex

    # complex coherence
    C = (x @ x.conj().T) / max(x.shape[1], 1)
    C = 0.5 * (C + C.conj().T)
    p = np.real(np.diag(C))
    p = np.maximum(p, 1e-18)
    G = C / np.sqrt(p[:, None] * p[None, :])

    plot_complex_matrix(G, title=f"Pairwise coherence from out['{use}']", show_phase=True,
                        prc_abs=(1, 99), mask_phase_below=mask_phase_below)
    plot_top_eigs(G, K=25, title=f"Eigenvalues of coherence (out['{use}'])")

    # phase-noise correlation (shot-to-shot phase increments)
    dphi = np.angle(x[:, 1:] * np.conj(x[:, :-1]))  # (M,N-1) real
    dphi = dphi - dphi.mean(axis=1, keepdims=True)
    dphi = dphi / (dphi.std(axis=1, keepdims=True) + 1e-12)
    Rphi = (dphi @ dphi.T) / max(dphi.shape[1], 1)
    Rphi = 0.5 * (Rphi + Rphi.T)
    np.fill_diagonal(Rphi, 1.0)

    plot_real_matrix(Rphi, title=f"Phase-increment correlation (out['{use}'])", symmetric=True)
    plot_top_eigs(Rphi, K=25, title=f"Eigenvalues of phase-increment corr (out['{use}'])")

    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def phase_increments(x_mn, lag=1, amp_mask=None, eps=1e-12):
    """
    Compute per-NV phase increments:
        dphi[n,t] = angle( x[n,t+lag] * conj(x[n,t]) )

    x_mn: (M,N) complex
    lag: integer >= 1
    amp_mask: None OR float threshold. If provided, mask times where |x| is too small:
              keeps only where |x[t]| and |x[t+lag]| >= amp_mask
    Returns dphi: (M, N-lag) float (may contain NaNs if masked)
    """
    x = np.asarray(x_mn, np.complex128)
    if lag < 1 or lag >= x.shape[1]:
        raise ValueError("lag must be >=1 and < Nshots")

    x0 = x[:, :-lag]
    x1 = x[:, lag:]

    dphi = np.angle(x1 * np.conj(x0))  # (M, N-lag)

    if amp_mask is not None:
        a0 = np.abs(x0)
        a1 = np.abs(x1)
        good = (a0 >= float(amp_mask)) & (a1 >= float(amp_mask)) & np.isfinite(dphi)
        dphi = np.where(good, dphi, np.nan)

    return dphi


def pearson_corr_matrix_from_rows(Y_mk, eps=1e-12):
    """
    Pearson correlation across rows of Y (M,K), allowing NaNs (pairwise complete).
    Returns R: (M,M) with diag=1.
    """
    Y = np.asarray(Y_mk, float)
    M, K = Y.shape
    R = np.full((M, M), np.nan, float)

    # Fast path if no NaNs:
    if np.isfinite(Y).all():
        Yc = Y - Y.mean(axis=1, keepdims=True)
        Ys = Yc / (Yc.std(axis=1, keepdims=True) + eps)
        R = (Ys @ Ys.T) / max(K - 1, 1)
        R = 0.5 * (R + R.T)
        np.fill_diagonal(R, 1.0)
        return R

    # Pairwise NaN-safe path:
    for i in range(M):
        yi = Y[i]
        mi = np.isfinite(yi)
        for j in range(i, M):
            yj = Y[j]
            m = mi & np.isfinite(yj)
            n = int(m.sum())
            if n < 3:
                continue
            a = yi[m]
            b = yj[m]
            a = a - a.mean()
            b = b - b.mean()
            denom = (np.sqrt(np.mean(a*a)) * np.sqrt(np.mean(b*b)) + eps)
            r = float(np.mean(a*b) / denom)
            R[i, j] = r
            R[j, i] = r
    np.fill_diagonal(R, 1.0)
    return R


# def phase_increment_pearson_matrix(x_mn, lag=1, amp_mask=None, eps=1e-12):
#     """
#     Convenience wrapper: returns (Rphi, dphi)
#     """
#     dphi = phase_increments(x_mn, lag=lag, amp_mask=amp_mask, eps=eps)
#     Rphi = pearson_corr_matrix_from_rows(dphi, eps=eps)
#     return Rphi, dphi

import numpy as np
import matplotlib.pyplot as plt


# ---------- utilities ----------
def _safe_percentile_vlim(A, prc=99):
    """Robust symmetric limits for plotting off-diagonal values."""
    A = np.array(A, float, copy=True)
    np.fill_diagonal(A, np.nan)
    v = np.nanpercentile(np.abs(A), prc)
    if not np.isfinite(v) or v <= 0:
        v = np.nanmax(np.abs(A))
        if not np.isfinite(v) or v <= 0:
            v = 1.0
    return float(v)

def offdiag_values(C, mode="abs"):
    """
    Return 1D array of off-diagonal entries.
    mode: 'abs', 'real', 'imag', 'phase'
    """
    C = np.asarray(C)
    M = C.shape[0]
    iu = np.triu_indices(M, k=1)

    if mode == "abs":
        v = np.abs(C[iu])
    elif mode == "real":
        v = np.real(C[iu])
    elif mode == "imag":
        v = np.imag(C[iu])
    elif mode == "phase":
        v = np.angle(C[iu])
    else:
        raise ValueError("mode must be abs/real/imag/phase")

    v = np.asarray(v).ravel()
    v = v[np.isfinite(v)]
    return v


# ---------- grouping / permutations ----------
def permutation_by_score(score, descending=True):
    score = np.asarray(score, float)
    order = np.argsort(score)
    if descending:
        order = order[::-1]
    groups = {"sorted": (0, len(order))}
    return order.astype(int), groups

def permutation_by_orientation(M, ORI_11m1, ORI_m111, keep_rest=True):
    ORI_11m1 = [i for i in ORI_11m1 if 0 <= i < M]
    ORI_m111 = [i for i in ORI_m111 if 0 <= i < M and i not in set(ORI_11m1)]
    perm = ORI_11m1 + ORI_m111
    if keep_rest:
        rest = [i for i in range(M) if i not in set(perm)]
        perm += rest
    perm = np.array(perm, int)
    n1 = len(ORI_11m1)
    n2 = len(ORI_m111)
    groups = {
        "ORI_11m1": (0, n1),
        "ORI_m111": (n1, n1 + n2),
        "rest": (n1 + n2, len(perm)),
    }
    return perm, groups

def split_by_sign_of_mode(v, frac_clip=0.0):
    """
    Split NVs into positive/negative groups based on a *mode loading* vector v.

    Handles complex v by fixing a global phase so sum(v) is real positive,
    then splitting by sign of real(v).

    frac_clip: optionally drop near-zero values (e.g. 0.02 keeps only |x| above 2% quantile).
    Returns (perm, groups, sign_value)
    """
    v = np.asarray(v, np.complex128)

    # Fix global phase so sum(v) becomes real positive (remove arbitrary eigenvector phase)
    s = np.sum(v)
    if np.abs(s) > 0:
        v = v * np.exp(-1j * np.angle(s))

    x = np.real(v)  # sign decision variable

    if frac_clip > 0:
        thr = np.quantile(np.abs(x), frac_clip)
        keep = np.abs(x) >= thr
    else:
        keep = np.ones_like(x, dtype=bool)

    pos = np.where((x >= 0) & keep)[0].tolist()
    neg = np.where((x < 0) & keep)[0].tolist()
    drop = np.where(~keep)[0].tolist()

    perm = np.array(pos + neg + drop, int)
    groups = {"pos": (0, len(pos)), "neg": (len(pos), len(pos) + len(neg)), "dropped": (len(pos) + len(neg), len(perm))}
    return perm, groups, x


# ---------- plotting ----------
def plot_matrix_with_groups(C, perm=None, groups=None, mode="abs", title="Matrix",
                            vlim=None, diag_color="0.85", prc=99):
    """
    mode:
      - for complex C: 'abs' or 'phase' or 'real' or 'imag'
      - for real C: 'real' is typical
    """
    C = np.asarray(C)
    M = C.shape[0]
    if perm is None:
        perm = np.arange(M, dtype=int)
    Cp = C[np.ix_(perm, perm)]

    # choose what to show
    if mode == "abs":
        A = np.abs(Cp)
    elif mode == "phase":
        A = np.angle(Cp)
    elif mode == "real":
        A = np.real(Cp)
    elif mode == "imag":
        A = np.imag(Cp)
    else:
        raise ValueError("mode must be abs/phase/real/imag")

    # mask diagonal (gray)
    mask = np.eye(M, dtype=bool)
    Am = np.ma.array(A, mask=mask)

    # set limits
    if vlim is None:
        if mode == "phase":
            vmin, vmax = -np.pi, np.pi
        else:
            v = _safe_percentile_vlim(A, prc=prc)
            vmin, vmax = (-v, v) if mode in ("real", "imag") else (0, v)
    else:
        if isinstance(vlim, (tuple, list)) and len(vlim) == 2:
            vmin, vmax = float(vlim[0]), float(vlim[1])
        else:
            v = float(vlim)
            vmin, vmax = (-v, v) if mode in ("real", "imag") else (0, v)

    cmap = plt.cm.viridis.copy() if mode in ("abs",) else plt.cm.RdBu_r.copy()
    cmap.set_bad(diag_color)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(Am, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("NV index (sorted)" if perm is not None else "NV index")
    ax.set_ylabel("NV index (sorted)" if perm is not None else "NV index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # boundaries + labels
    if groups is not None:
        for name, (a, b) in groups.items():
            if a > 0:
                ax.axhline(a - 0.5, color="w", lw=1.2)
                ax.axvline(a - 0.5, color="w", lw=1.2)
        for name, (a, b) in groups.items():
            if (b - a) <= 0:
                continue
            c = 0.5 * (a + b - 1)
            ax.text(-0.02, c, name, va="center", ha="right", transform=ax.get_yaxis_transform())
            ax.text(c, -0.03, name, va="top", ha="center", transform=ax.get_xaxis_transform())

    ax.grid(False)
    plt.tight_layout()
    return fig, ax


def plot_offdiag_hist(C, mode="abs", bins=80, title="Off-diagonal histogram"):
    v = offdiag_values(C, mode=mode)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.hist(v, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(f"off-diagonal {mode}(C_ij)")
    ax.set_ylabel("count")
    ax.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    return fig, ax


def plot_offdiag_hist_within_cross(C, groupA, groupB, mode="abs", bins=80,
                                  title="Within vs Cross hist"):
    C = np.asarray(C)
    A = np.array(groupA, int)
    B = np.array(groupB, int)

    def _vals(idx1, idx2, upper_only=True):
        sub = C[np.ix_(idx1, idx2)]
        if idx1 is idx2 and upper_only:
            m = np.triu(np.ones(sub.shape, bool), k=1)
            sub = sub[m]
        else:
            sub = sub.ravel()
        if mode == "abs": sub = np.abs(sub)
        elif mode == "real": sub = np.real(sub)
        elif mode == "imag": sub = np.imag(sub)
        elif mode == "phase": sub = np.angle(sub)
        sub = sub[np.isfinite(sub)]
        return sub

    vAA = _vals(A, A, upper_only=True)
    vBB = _vals(B, B, upper_only=True)
    vAB = _vals(A, B, upper_only=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.hist(vAA, bins=bins, alpha=0.6, label="within A")
    ax.hist(vBB, bins=bins, alpha=0.6, label="within B")
    ax.hist(vAB, bins=bins, alpha=0.6, label="cross A×B")
    ax.set_title(title)
    ax.set_xlabel(f"{mode}(C_ij)")
    ax.set_ylabel("count")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax


def phase_increment_series(x_mn, lag=1, amp_mode="min"):
    """
    x_mn: (M,N) complex series
    Returns:
      dphi: (M, N-lag) in (-pi, pi]
      amp:  (M, N-lag) amplitude proxy used for masking
    """
    x = np.asarray(x_mn, np.complex128)
    if lag < 1 or lag >= x.shape[1]:
        raise ValueError("lag must satisfy 1 <= lag < N")

    x1 = x[:, lag:]
    x0 = x[:, :-lag]

    d = x1 * np.conj(x0)
    dphi = np.angle(d)

    a1 = np.abs(x1)
    a0 = np.abs(x0)
    if amp_mode == "min":
        amp = np.minimum(a0, a1)
    elif amp_mode == "geom":
        amp = np.sqrt(a0 * a1)
    else:
        raise ValueError("amp_mode must be 'min' or 'geom'")

    return dphi, amp


def phase_increment_pearson_matrix(
    x_mn,
    lag=1,
    amp_mask=None,
    amp_mode="min",
    eps=1e-12,
    return_aux=False,
):
    """
    Pearson corr across NVs of phase increments dphi.

    amp_mask options:
      - None: no mask
      - scalar thr: keep samples where amp >= thr (global threshold)
      - array shape (M,) or (M,1): per-NV threshold

    Returns:
      Rphi: (M,M) real, diag = 1
      dphi: (M, N-lag) phase increments
      (optional aux): dict with mask stats
    """
    dphi, amp = phase_increment_series(x_mn, lag=lag, amp_mode=amp_mode)
    M, T = dphi.shape

    # build validity mask W (M,T)
    if amp_mask is None:
        W = np.ones((M, T), dtype=bool)
    else:
        thr = np.asarray(amp_mask)
        if thr.ndim == 0:   # scalar
            W = amp >= float(thr)
        else:
            thr = thr.reshape(M, 1)
            W = amp >= thr

    # center per NV over its valid samples
    X = dphi.astype(np.float32)
    Wf = W.astype(np.float32)

    n_i = np.sum(Wf, axis=1)  # (M,)
    n_i_safe = np.maximum(n_i, 1.0)

    mu = (X * Wf).sum(axis=1) / n_i_safe
    Xc = (X - mu[:, None]) * Wf

    # pairwise counts (intersection sizes) and dot products
    Nij = Wf @ Wf.T                      # (M,M)
    Sij = Xc @ Xc.T                      # (M,M)

    # per-NV variance estimates
    var_i = np.diag(Sij) / np.maximum(n_i - 1.0, 1.0)
    var_i = np.maximum(var_i, eps)

    denom = np.sqrt(var_i[:, None] * var_i[None, :])

    # covariance uses Nij; correlation uses per-NV vars
    cov = Sij / np.maximum(Nij - 1.0, 1.0)
    Rphi = cov / denom

    # clean up / clip
    Rphi = np.clip(Rphi, -1.0, 1.0)
    np.fill_diagonal(Rphi, 1.0)

    if not return_aux:
        return Rphi, dphi
    aux = {
        "W": W,
        "Nij": Nij,
        "n_i": n_i,
        "mu": mu,
        "amp_used": amp,
    }
    return Rphi, dphi, aux

def plot_matrix_views(
    A, title,
    ORI_11m1=None, ORI_m111=None,
    sign_vec=None,
    score=None,
    bins=80,
    show_phase=False,
):
    """
    A: (M,M) real or complex matrix
    score: (M,) per-NV score for sorting (optional)
    sign_vec: (M,) vector to split pos/neg (optional; e.g. v1 of C_time)
    """

    M = A.shape[0]

    # 1) raw
    plot_matrix_with_groups(A, mode=("abs" if np.iscomplexobj(A) else "real"), title=title + " (raw)")
    plot_offdiag_hist(A, mode=("abs" if np.iscomplexobj(A) else "real"), bins=bins,
                      title=title + " (offdiag hist)")

    # 2) sorted by score (e.g. |v1| or scoreK)
    if score is not None:
        permS, groupsS = permutation_by_score(score, descending=True)
        plot_matrix_with_groups(A, perm=permS, groups=groupsS,
                                mode=("abs" if np.iscomplexobj(A) else "real"),
                                title=title + " (sorted by score)")
        plot_offdiag_hist(A[np.ix_(permS, permS)],
                          mode=("abs" if np.iscomplexobj(A) else "real"),
                          bins=bins,
                          title=title + " (sorted offdiag hist)")

    # 3) orientation grouping
    if ORI_11m1 is not None and ORI_m111 is not None:
        permO, groupsO = permutation_by_orientation(M, ORI_11m1, ORI_m111, keep_rest=True)
        plot_matrix_with_groups(A, perm=permO, groups=groupsO,
                                mode=("abs" if np.iscomplexobj(A) else "real"),
                                title=title + " (orientation grouped)")
        plot_offdiag_hist_within_cross(A, ORI_11m1, ORI_m111,
                                       mode=("abs" if np.iscomplexobj(A) else "real"),
                                       bins=bins,
                                       title=title + " (within/cross by orientation)")

    # 4) split into +/- groups using sign_vec
    if sign_vec is not None:
        permP, groupsP, xsign = split_by_sign_of_mode(sign_vec, frac_clip=0.0)
        plot_matrix_with_groups(A, perm=permP, groups=groupsP,
                                mode=("abs" if np.iscomplexobj(A) else "real"),
                                title=title + " (split by +/- group)")
        # hist within/cross for pos/neg
        pos_idx = permP[np.arange(groupsP["pos"][0], groupsP["pos"][1])]
        neg_idx = permP[np.arange(groupsP["neg"][0], groupsP["neg"][1])]
        plot_offdiag_hist_within_cross(A, pos_idx, neg_idx,
                                       mode=("abs" if np.iscomplexobj(A) else "real"),
                                       bins=bins,
                                       title=title + " (within/cross pos-neg)")

    plt.show()

import numpy as np

def perm_by_spectral_seriation(A, use_abs=True, eps=1e-12):
    """
    Returns a permutation that tends to place highly correlated indices adjacent.
    Uses Fiedler vector of graph Laplacian of similarity matrix.

    A: (M,M) real/complex
    use_abs: if True, uses S=|A| as similarity. Otherwise uses S=A (clipped).
    """
    A = np.asarray(A)
    if np.iscomplexobj(A):
        S = np.abs(A) if use_abs else np.real(A)
    else:
        S = np.abs(A) if use_abs else A.copy()

    # remove diagonal self-similarity
    S = S.copy()
    np.fill_diagonal(S, 0.0)

    # ensure nonnegative
    S = np.maximum(S, 0.0)

    # Laplacian
    d = S.sum(axis=1)
    L = np.diag(d) - S

    # eigenvectors of symmetric L
    w, V = np.linalg.eigh(0.5 * (L + L.T))

    # Fiedler vector = 2nd smallest eigenvector (skip constant mode)
    if V.shape[1] < 2:
        return np.arange(A.shape[0], dtype=int)

    v = V[:, 1]
    perm = np.argsort(v)
    return perm


def plot_orientation_energy(out, ori_idx, dt=1.0, nshots=None, zthr=20.0, title=""):
    s = np.asarray(out["s0"])  # (M,N)
    ori_idx = np.array(ori_idx, dtype=int)
    ori_idx = ori_idx[(ori_idx >= 0) & (ori_idx < s.shape[0])]
    if len(ori_idx) == 0:
        print(f"{title}: empty orientation list")
        return

    x = s[ori_idx, :]
    if nshots is not None:
        nshots = min(int(nshots), x.shape[1])
        x = x[:, :nshots]

    # Energy across NVs in this orientation
    E = np.nanmean(np.abs(x)**2, axis=0)

    # Robust z-score for glitch marking
    zE, med, sig = robust_zscore_mad(E)
    glitch = np.abs(zE) >= float(zthr)

    t = np.arange(E.size) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    plt.figure(figsize=(11, 3.0))
    plt.plot(t, E, lw=1)
    if np.any(glitch):
        plt.plot(t[glitch], E[glitch], "rx", ms=4, label=f"|z|≥{zthr:g}")
        plt.legend(fontsize=8, loc="upper right")
    plt.xlabel(xlab)
    plt.ylabel(r"$\langle |s_0|^2\rangle_{NV}$")
    plt.title(title)
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()


def plot_orientation_energy_and_phase_coherence(
    out, ori_idx, dt=1.0, nshots=None, amp_prc=1.0, title=""
):
    """
    Panels:
      (1) Energy E[k] = <|s0|^2> over NVs in this orientation
      (2) Circular-mean phase increment dphi_bar[k] and coherence R[k]
          using per-NV dphi_n[k] = arg(s[n,k] * conj(s[n,k-1])).

    amp_prc: per-NV amplitude percentile used to mask low-amp shots (noise).
    """
    s = np.asarray(out["s0"])  # (M,N)
    ori_idx = np.asarray(ori_idx, dtype=int)
    ori_idx = ori_idx[(ori_idx >= 0) & (ori_idx < s.shape[0])]
    if ori_idx.size == 0:
        print(f"{title}: empty orientation list")
        return

    x = s[ori_idx, :]  # (Mori, N)
    if nshots is not None:
        nshots = min(int(nshots), x.shape[1])
        x = x[:, :nshots]

    Mori, N = x.shape
    t = np.arange(N, dtype=float) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    # (1) Energy (amplitude-only)
    E = np.nanmean(np.abs(x)**2, axis=0)

    # (2) Per-NV phase increment
    # dphi_n[k] uses adjacent points, so length N-1
    dphi = np.angle(x[:, 1:] * np.conj(x[:, :-1]))  # (Mori, N-1)

    # Mask dphi when amplitude is small for that NV (noise-dominated)
    amp = np.abs(x)  # (Mori, N)
    thr_nv = np.nanpercentile(amp, amp_prc, axis=1)  # (Mori,)
    good = (amp[:, 1:] > thr_nv[:, None]) & (amp[:, :-1] > thr_nv[:, None])
    dphi_m = np.where(good, dphi, np.nan)

    # Circular mean + coherence across NVs
    z = np.exp(1j * dphi_m)  # (Mori, N-1)
    z_mean = np.nanmean(z, axis=0)  # (N-1,)
    dphi_bar = np.angle(z_mean)     # circular mean step
    R = np.abs(z_mean)              # 0..1 coherence (correlated-jump detector)

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)

    ax[0].plot(t, E, lw=1)
    ax[0].set_ylabel(r"$\langle |s_0|^2\rangle$")
    ax[0].set_title(title + " — energy (amplitude-only)")
    ax[0].grid(True, ls="--", lw=0.5)

    ax[1].plot(t[1:], dphi_bar, lw=1, label=r"$\bar{\Delta\phi}$ (circular mean)")
    # ax2 = ax[1].twinx()
    # ax2.plot(t[1:], R, lw=1, alpha=0.9, label="R (coherence)")

    # ax[1].set_ylabel(r"$\bar{\Delta\phi}$ (rad)")
    # ax2.set_ylabel("R (0..1)")
    # ax[1].set_xlabel(xlab)
    # ax[1].set_title("Per-NV phase increment: mean step + coherence R")
    # ax[1].grid(True, ls="--", lw=0.5)

    # Combined legend
    lines, labels = ax[1].get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax[1].legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")

    plt.tight_layout()

    return {"t": t, "E": E, "dphi_bar": dphi_bar, "R": R, "mask_good_frac": np.nanmean(good)}


import numpy as np
import matplotlib.pyplot as plt

def _apply_keep_runs_to_counts(counts, keep_runs):
    """counts: (4,M,R,S,P). keep_runs can be None or array of run indices."""
    if keep_runs is None:
        return counts
    keep_runs = np.asarray(keep_runs, dtype=int)
    return counts[:, :, keep_runs, :, :]

def orientation_brightness_series(counts, out, ori_idx, nshots=None):
    """
    Returns B[k] for an orientation group using raw counts:
      B_n[k] = Ip+Im+Qp+Qm, then average across NVs in ori_idx.
    Aligned to out['s0'] by applying out['keep_runs'] and out['keep_shots'].
    """
    counts = np.asarray(counts)
    assert counts.ndim == 5, f"counts must be (4,M,R,S,P), got {counts.shape}"
    assert counts.shape[0] == 4, "counts first axis must be 4 (Ip,Im,Qp,Qm)"

    # Apply same run filtering as analyze_counts_fft_corr
    counts_used = _apply_keep_runs_to_counts(counts, out.get("keep_runs", None))
    _, M, R, S, P = counts_used.shape
    Ntot = R * S * P

    ori_idx = np.asarray(ori_idx, dtype=int)
    ori_idx = ori_idx[(ori_idx >= 0) & (ori_idx < M)]
    if ori_idx.size == 0:
        raise ValueError("Empty ori_idx after bounds check.")

    # Flatten shots to match out['keep_shots'] length
    c = counts_used.reshape(4, M, Ntot)          # (4,M,Ntot)
    B_nv = np.nansum(c, axis=0)                  # (M,Ntot) sum over 4 frames
    B_ori = np.nanmean(B_nv[ori_idx, :], axis=0) # (Ntot,)

    # Align with out['s0'] shots (drop shots that were dropped in lock-in)
    keep_shots = out.get("keep_shots", None)
    if keep_shots is not None:
        keep_shots = np.asarray(keep_shots, dtype=bool)
        if keep_shots.size != Ntot:
            raise ValueError(f"keep_shots length {keep_shots.size} != Ntot {Ntot}")
        B_ori = B_ori[keep_shots]

    if nshots is not None:
        B_ori = B_ori[:min(int(nshots), B_ori.size)]

    return B_ori

def orientation_energy_series(out, ori_idx, nshots=None):
    """
    E[k] = mean_{NV in ori} |s0|^2 using out['s0'] (already aligned shots).
    """
    s0 = np.asarray(out["s0"])  # (M,N)
    M, N = s0.shape
    ori_idx = np.asarray(ori_idx, dtype=int)
    ori_idx = ori_idx[(ori_idx >= 0) & (ori_idx < M)]
    if ori_idx.size == 0:
        raise ValueError("Empty ori_idx after bounds check.")

    s = s0[ori_idx, :]
    if nshots is not None:
        s = s[:, :min(int(nshots), s.shape[1])]

    E = np.nanmean(np.abs(s)**2, axis=0)
    return E

def shade_windows(ax, windows, dt=1.0):
    """
    windows: list of (t0,t1) in seconds if dt != 1, else in shot index units.
    """
    if windows is None:
        return
    for (t0, t1) in windows:
        ax.axvspan(t0, t1, alpha=0.15)

def plot_brightness_vs_energy(
    out, counts, ori_idx, dt=1.0, nshots=None, title="", windows=None
):
    """
    Two-panel plot: Brightness proxy B[k] vs Lock-in energy E[k] for one orientation group.
    """
    B = orientation_brightness_series(counts, out, ori_idx, nshots=nshots)
    E = orientation_energy_series(out, ori_idx, nshots=nshots)

    N = min(B.size, E.size)
    B = B[:N]
    E = E[:N]
    t = np.arange(N) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    # correlation (helps quantify “amplitude-dominated”)
    Bz = (B - np.nanmean(B)) / (np.nanstd(B) + 1e-12)
    Ez = (E - np.nanmean(E)) / (np.nanstd(E) + 1e-12)
    corr = np.nanmean(Bz * Ez)

    fig, ax = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)

    ax[0].plot(t, B, lw=1)
    shade_windows(ax[0], windows)
    ax[0].set_ylabel("B = <Ip+Im+Qp+Qm>")
    ax[0].set_title(f"{title} — brightness proxy (raw counts)")

    ax[1].plot(t, E, lw=1)
    shade_windows(ax[1], windows)
    ax[1].set_ylabel(r"E = $\langle |s_0|^2\rangle$")
    ax[1].set_xlabel(xlab)
    ax[1].set_title(f"lock-in energy (corr(B,E)≈{corr:.3f})")

    for a in ax:
        a.grid(True, ls="--", lw=0.5)

    plt.tight_layout()
    plt.show()

    return {"t": t, "B": B, "E": E, "corr_BE": corr}

import numpy as np
import matplotlib.pyplot as plt

def _apply_keep_runs_to_counts(counts, keep_runs):
    if keep_runs is None:
        return counts
    keep_runs = np.asarray(keep_runs, dtype=int)
    return counts[:, :, keep_runs, :, :]

def plot_ori_sumdiff_contrast_and_E(out, counts, ori_idx, dt=1.0, nshots=None, title="", windows=None):
    """
    Plots for an orientation subset:
      B  = <Ip+Im+Qp+Qm>    (brightness proxy)
      DI = <Ip-Im>, DQ = <Qp-Qm>
      CI = DI/<Ip+Im>, CQ = DQ/<Qp+Qm>   (contrast proxies)
      E  = <|s0|^2> from out['s0']
    """
    counts = np.asarray(counts)  # (4,M,R,S,P)
    assert counts.shape[0] == 4, "counts must be ordered [Ip, Im, Qp, Qm] on axis 0"
    s0 = np.asarray(out["s0"])   # (M,Nshots_used)
    M = s0.shape[0]

    ori_idx = np.asarray(ori_idx, dtype=int)
    ori_idx = ori_idx[(ori_idx >= 0) & (ori_idx < M)]
    if ori_idx.size == 0:
        print(f"{title}: empty orientation list")
        return

    # align counts to out's run filtering
    counts_used = _apply_keep_runs_to_counts(counts, out.get("keep_runs", None))
    _, M2, R, S, P = counts_used.shape
    assert M2 == M, f"M mismatch: counts has {M2}, out['s0'] has {M}"
    Ntot = R * S * P

    c = counts_used.reshape(4, M, Ntot)  # (4,M,Ntot)

    keep_shots = out.get("keep_shots", None)
    if keep_shots is not None:
        keep_shots = np.asarray(keep_shots, dtype=bool)
        c = c[:, :, keep_shots]  # (4,M,N)
    N = c.shape[2]

    # apply nshots
    if nshots is not None:
        N = min(int(nshots), N)
        c = c[:, :, :N]

    Ip, Im, Qp, Qm = c[0], c[1], c[2], c[3]  # each (M,N)

    # orientation averages
    mIp = np.nanmean(Ip[ori_idx], axis=0)
    mIm = np.nanmean(Im[ori_idx], axis=0)
    mQp = np.nanmean(Qp[ori_idx], axis=0)
    mQm = np.nanmean(Qm[ori_idx], axis=0)

    B  = mIp + mIm + mQp + mQm
    DI = mIp - mIm
    DQ = mQp - mQm

    # contrast proxies
    CI = DI / (mIp + mIm + 1e-12)
    CQ = DQ / (mQp + mQm + 1e-12)

    # lock-in energy (already aligned shots)
    E = np.nanmean(np.abs(s0[ori_idx, :N])**2, axis=0)

    t = np.arange(N) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    def shade(ax):
        if windows is None: return
        for (t0, t1) in windows:
            ax.axvspan(t0, t1, alpha=0.15)

    fig, ax = plt.subplots(4, 1, figsize=(11, 7.2), sharex=True)

    # ax[0].plot(t, B, lw=1);  shade(ax[0])
    # ax[0].set_ylabel("Raw Co"); ax[0].set_title(title + " — brightness & differences")
    
    # raw windows (this is where charge/brightness shows up)
    ax[0].plot(t, mIp, lw=1, label="Ip")
    ax[0].plot(t, mIm, lw=1, label="Im")
    ax[0].plot(t, mQp, lw=1, label="Qp")
    ax[0].plot(t, mQm, lw=1, label="Qm")
    ax[0].set_title(title + " — Raw Counts")
    ax[0].set_ylabel("Raw Coutns")
    ax[0].legend(fontsize=8, loc="upper right")
    # shade(ax[0]); ax[0].legend(fontsize=8, loc="upper right")
    # ax[0].set_ylabel("windows" + ylab_suffix)
    ax[0].set_title(title)

    ax[1].plot(t, DI, lw=1, label="DI=<Ip-Im>")
    ax[1].plot(t, DQ, lw=1, label="DQ=<Qp-Qm>")
    # shade(ax[1]); 
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_ylabel("DI, DQ")

    ax[2].plot(t, CI, lw=1, label="CI=DI/<Ip+Im>")
    ax[2].plot(t, CQ, lw=1, label="CQ=DQ/<Qp+Qm>")
    # shade(ax[2]); 
    ax[2].legend(fontsize=8, loc="upper right")
    ax[2].set_ylabel("contrast")

    ax[3].plot(t, E, lw=1)
    # shade(ax[3])
    ax[3].set_ylabel(r"E=<|s0|^2>"); ax[3].set_xlabel(xlab)
    ax[3].set_title("lock-in energy")

    for a in ax:
        a.grid(True, ls="--", lw=0.5)

    plt.tight_layout()
    plt.show()

    # quick correlations to guide interpretation
    # def corr(a,b):
    #     a=(a-np.nanmean(a))/(np.nanstd(a)+1e-12)
    #     b=(b-np.nanmean(b))/(np.nanstd(b)+1e-12)
    #     return float(np.nanmean(a*b))

    # print("corr(E,B) =", corr(E,B))
    # print("corr(E,CI) =", corr(E,CI), " corr(E,CQ) =", corr(E,CQ))

    return {"t": t, "B": B, "DI": DI, "DQ": DQ, "CI": CI, "CQ": CQ, "E": E}

import numpy as np
import matplotlib.pyplot as plt

def _apply_keep_runs_to_counts(counts, keep_runs):
    if keep_runs is None:
        return counts
    keep_runs = np.asarray(keep_runs, dtype=int)
    return counts[:, :, keep_runs, :, :]

def plot_lockin_simple_ori_or_all(
    out,
    counts,
    ori_idx=None,     # None -> all NVs; else list of NV indices
    dt=1.0,
    nshots=None,
    title="",
    eps=1e-12,
):
    """
    Simple, no grouping:

    If ori_idx is None:
        compute averages over ALL NVs.
    Else:
        compute averages only over NVs in ori_idx.

    Definitions (per NV n, shot k):
      I_n[k] = Ip_n[k] - Im_n[k]
      Q_n[k] = Qp_n[k] - Qm_n[k]
      s_n[k] = I_n[k] + i Q_n[k]     (out['s0'][n,k])
      g[k]   = <s_n[k]>_n
      E[k]   = <|s_n[k]|^2>_n
      E_c[k] = |g[k]|^2
      E_v[k] = E - E_c = <|s-g|^2>_n
      phi[k]  = arg(g[k])
      dphi[k] = arg(g[k] g*[k-1])
      gamma[k] = E_c/(E+eps)
    """
    counts = np.asarray(counts)
    assert counts.ndim == 5 and counts.shape[0] == 4, "counts must be (4,M,R,S,P) ordered [Ip,Im,Qp,Qm]"
    s0 = np.asarray(out["s0"])  # (M,Nshots), complex
    M = s0.shape[0]

    # ---- choose NV subset ----
    if ori_idx is None:
        idx = np.arange(M, dtype=int)
        subset_name = "ALL NVs"
    else:
        idx = np.asarray(ori_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < M)]
        if idx.size == 0:
            raise ValueError("ori_idx is empty after sanitizing.")
        subset_name = f"ORI subset (N={idx.size})"

    # ---- Align counts to out filtering ----
    counts_used = _apply_keep_runs_to_counts(counts, out.get("keep_runs", None))
    _, M2, R, S, P = counts_used.shape
    assert M2 == M, f"M mismatch: counts has {M2}, out['s0'] has {M}"
    Ntot = R * S * P
    c = counts_used.reshape(4, M, Ntot)

    keep_shots = out.get("keep_shots", None)
    if keep_shots is not None:
        keep_shots = np.asarray(keep_shots, dtype=bool)
        if keep_shots.size != Ntot:
            raise ValueError(f"keep_shots length {keep_shots.size} != Ntot {Ntot}")
        c = c[:, :, keep_shots]  # (4,M,N)
    N = c.shape[2]

    if nshots is not None:
        N = min(int(nshots), N)
        c = c[:, :, :N]
    N = c.shape[2]

    Ip, Im, Qp, Qm = c[0], c[1], c[2], c[3]  # (M,N)

    # ---- means over chosen NVs ----
    mIp = np.nanmean(Ip[idx], axis=0)
    mIm = np.nanmean(Im[idx], axis=0)
    mQp = np.nanmean(Qp[idx], axis=0)
    mQm = np.nanmean(Qm[idx], axis=0)

    DI = mIp - mIm
    DQ = mQp - mQm

    # ---- lock-in phasors for chosen NVs ----
    x = s0[idx, :N]                      # (Nnv,N) complex
    g = np.nanmean(x, axis=0)            # <s>
    E = np.nanmean(np.abs(x)**2, axis=0) # <|s|^2>
    E_common = np.abs(g)**2
    E_var = np.maximum(E - E_common, 0.0)
    gamma = E_common / (E + eps)

    phi = np.angle(g)
    dphi = np.angle(g[1:] * np.conj(g[:-1]))

    t = np.arange(N, dtype=float) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    # ---- plot ----
    fig, ax = plt.subplots(5, 1, figsize=(12, 8.8), sharex=True)

    ax[0].plot(t, mIp, lw=1, label="Ip")
    ax[0].plot(t, mIm, lw=1, label="Im")
    ax[0].plot(t, mQp, lw=1, label="Qp")
    ax[0].plot(t, mQm, lw=1, label="Qm")
    ax[0].set_ylabel("raw counts")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"{title} — {subset_name}" if title else subset_name)
    ax[0].text(0.01, 0.98, r"Raw: $I^+,I^-,Q^+,Q^-$ (interleaves)", transform=ax[0].transAxes, va="top")

    ax[1].plot(t, DI, lw=1, label="DI=<Ip-Im>")
    ax[1].plot(t, DQ, lw=1, label="DQ=<Qp-Qm>")
    ax[1].set_ylabel("DI, DQ")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].text(0.01, 0.98, r"$I=I^+-I^-,\ Q=Q^+-Q^-$ (from raw means)", transform=ax[1].transAxes, va="top")

    # ax[2].plot(t, E, lw=1, label=r"$E=\langle|s|^2\rangle$")
    # ax[2].plot(t, E_common, lw=1, label=r"$E_c=|\langle s\rangle|^2$")
    # ax[2].plot(t, E_var, lw=1, label=r"$E_v=E-E_c$")
    # ax[2].set_ylabel("energy")
    # ax[2].legend(fontsize=8, loc="upper right")
    # ax[2].text(0.01, 0.98, r"$s=I+iQ$; $E=\langle|s|^2\rangle$; $E=E_c+E_v$", transform=ax[2].transAxes, va="top")

    ax[3].plot(t, gamma, lw=1, label=r"$\gamma=E_c/(E+\epsilon)$")
    ax[3].plot(t, np.abs(g), lw=1, label=r"$|\langle s\rangle|$")
    ax[3].plot(t, np.sqrt(np.maximum(E, 0.0)), lw=1, label=r"$\sqrt{E}$")
    ax[3].set_ylabel("coh/amp")
    ax[3].set_ylim(-0.05, 0.5)
    ax[3].legend(fontsize=8, loc="upper right")

    # ax[4].plot(t, phi, lw=1, label=r"$\phi=\arg\langle s\rangle$")
    # ax[4].plot(t[1:], dphi, lw=1, label=r"$\Delta\phi=\arg(g_{k}g^{*}_{k-1})$")
    # ax[4].set_ylabel("phase (rad)")
    # ax[4].set_xlabel(xlab)
    # ax[4].legend(fontsize=8, loc="upper right")

    for a in ax:
        a.grid(True, ls="--", lw=0.5)

    plt.tight_layout()
    plt.show(block=False)

    return {
        "t": t,
        "idx": idx,
        "mIp": mIp, "mIm": mIm, "mQp": mQp, "mQm": mQm,
        "DI": DI, "DQ": DQ,
        "E": E, "E_common": E_common, "E_var": E_var,
        "g": g, "phi": phi, "dphi": dphi, "gamma": gamma,
    }


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_lockin_simple(
    out,
    counts,
    ori_idx=None,     # None -> all NVs; else list of NV indices
    dt=1.0,
    nshots=None,
    title="",
    seq=r"$\pi/2_x — \ \pi_x \ — \pi/2_{\{+x,-x,+y,-y\}}$",
    eps=1e-12,
):
    counts = np.asarray(counts)
    assert counts.ndim == 5 and counts.shape[0] == 4, "counts must be (4,M,R,S,P) ordered [Ip,Im,Qp,Qm]"

    _, M, R, S, P = counts.shape

    if ori_idx is None:
        idx = np.arange(M, dtype=int)
        subset_name = "ALL NVs"
    else:
        idx = np.asarray(ori_idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < M)]
        if idx.size == 0:
            raise ValueError("ori_idx is empty after sanitizing.")
        subset_name = f"ORI subset (N={idx.size})"

    Ntot = R * S * P
    c = counts.reshape(4, M, Ntot)

    N = c.shape[2]
    if nshots is not None:
        N = min(int(nshots), N)
        c = c[:, :, :N]
    N = c.shape[2]

    Ip, Im, Qp, Qm = c[0], c[1], c[2], c[3]

    t = np.arange(N, dtype=float) * float(dt)
    xlab = "time (s)" if dt != 1.0 else "shot index"

    for nv in idx:
        fig, ax = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)

        # Label ONLY by final pulse axis (no I/Q wording)
        ax[0].plot(t, Ip[nv, :N], lw=1, label=r"final $\pi/2_{+x}$")
        ax[1].plot(t, Im[nv, :N], lw=1, label=r"final $\pi/2_{-x}$")
        ax[2].plot(t, Qp[nv, :N], lw=1, label=r"final $\pi/2_{+y}$")
        ax[3].plot(t, Qm[nv, :N], lw=1, label=r"final $\pi/2_{-y}$")

        ax[0].set_ylabel(r"$\pi/2_{+x}$")
        ax[1].set_ylabel(r"$\pi/2_{-x}$")
        ax[2].set_ylabel(r"$\pi/2_{+y}$")
        ax[3].set_ylabel(r"$\pi/2_{-y}$")
        ax[3].set_xlabel(xlab)

        base = f"{title} — " if title else ""
        ax[0].set_title(f"{base}{subset_name} — NV idx: {nv}  |  seq: {seq}")

        ax[0].text(
            0.01, 0.98,
            r"Interleaves correspond to final pulse: $+x,-x,+y,-y$",
            transform=ax[0].transAxes, va="top"
        )

        for a in ax:
            a.grid(True, ls="--", lw=0.5)
            a.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.show()

    return {"t": t, "idx": idx}


# ----------------------------
# Example usage (edit for your loader)
# ----------------------------
if __name__ == "__main__":
    # Example: load from an npz that contains 'counts'
    from utils import data_manager as dm
    # raw_data = dm.get_raw_data(
    #     # file_stem="2025_12_24-09_32_29-johnson-nv0_2025_10_21", load_npz=True
    # )
    file_stems = ["2026_01_04-18_43_01-johnson-nv0_2025_10_21",
                  "2026_01_05-14_32_43-johnson-nv0_2025_10_21"]
    # raw_data = dm.get_raw_data(
    #     file_stem="2026_01_04-18_43_01-johnson-nv0_2025_10_21", load_npz=True
    # )
    raw_data= widefield.process_multiple_files(file_stems, load_npz=True)

    nv_list =  raw_data["nv_list"] 
    counts = raw_data["counts"]
    
    out = analyze_counts_fft_corr(
    counts,
    dt=0.06,                 # set to real seconds/shot if you know it
    global_bad_thresh=0.9,
    trim_trailing=True,
    drop_all_bad_runs=False,
    detrend=False,
    regress_global=False,
    regress_use_Q=False,
    fband=None,
    )
    # #fmt:off 
    ORI_11m1 = [0, 1, 3, 5, 6, 7, 9, 10, 13, 18, 19, 21, 24, 25, 27, 28, 30, 32, 34, 36, 40, 41, 43, 44, 46, 48, 49, 51, 52, 53, 56, 57, 64, 65, 66, 67, 68, 69, 73, 75, 77, 80, 82, 84, 86, 88, 91, 98, 100, 101, 102, 103, 106, 107, 109, 110, 111, 113, 115, 116, 118, 119, 120, 121, 123, 124, 127, 129, 130, 131, 132, 133, 134, 135, 141, 142, 146, 149, 150, 152, 153, 156, 157, 158, 162, 163, 165, 167, 168, 171, 174, 177, 179, 184, 185, 186, 187, 189, 190, 191, 192, 193, 195, 198, 201, 203]
    ORI_m111 = [2, 4, 8, 11, 12, 14, 15, 16, 17, 20, 22, 23, 26, 29, 31, 33, 35, 37, 38, 39, 42, 45, 47, 50, 54, 55, 58, 59, 60, 61, 62, 63, 70, 71, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 90, 92, 93, 94, 95, 96, 97, 99, 104, 105, 108, 112, 114, 117, 122, 125, 126, 128, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 151, 154, 155, 159, 160, 161, 164, 166, 169, 170, 172, 173, 175, 176, 178, 180, 181, 182, 183, 188, 194, 196, 197, 199, 200, 202] 
    # ORI_m111_ORI_11m1 = np.sort1(ORI_11m1 + ORI_m111)
    ORI_m111_ORI_11m1 = np.unique(ORI_11m1 + ORI_m111)

    
    # # #fmt:on
    # make_summary_plots(out, dt=0.240, global_bad_thresh=0.9, zthr=10.0)
    # plot_shot_diagnostics(out, use="s0", nshots=300000)  # most “physical”
    # plot_shot_diagnostics(out, use="z",  nshots=300000)  # ensemble outlier score
    # res = plot_physical_shot_diagnostics_s0(
    # out,
    # nshots=None,   # or None for all shots
    # dt=0.248,        # seconds per complex shot (your value)
    # zthr=20.0,       # glitch threshold in robust z-score units
    # amp_prc=20.0,   # mask phase-increment when |g| is in lowest 20%
    # mark_glitches=False)
    
    # --- Use it ---
    # dt = 0.248
    # plot_orientation_energy(out, ORI_11m1, dt=dt, zthr=20, title="ORI_11m1: shot energy")
    # plot_orientation_energy(out, ORI_m111, dt=dt, zthr=20, title="ORI_m111: shot energy")
    # --- Use it ---
    # dt = 0.248
    # plot_orientation_energy_and_phase_coherence(out, ORI_11m1, dt=dt, amp_prc=20, title="ORI_11m1")
    # plot_orientation_energy_and_phase_coherence(out, ORI_m111, dt=dt, amp_prc=20, title="ORI_m111")
    # plt.show()
    # plt.show()

    # dt = 0.248
    # dt = 1.0
    # windows = [(29000, 36000), (69000, 72000)]  # seconds (since dt is seconds/shot)

    # plot_brightness_vs_energy(out, counts, ORI_11m1, dt=dt, title="ORI_11m1", windows=windows)
    # plot_brightness_vs_energy(out, counts, ORI_m111, dt=dt, title="ORI_m111", windows=windows)

    # plot_ori_sumdiff_contrast_and_E(out, counts, ORI_m111_ORI_11m1, dt=dt, title="ORI_11m1", windows=windows)
    # plot_ori_sumdiff_contrast_and_E(out, counts, ORI_m111_ORI_11m1, dt=dt, title="ORI_m111", windows=windows)

    # plot_ori_debug_decomposed(out, counts, ORI_11m1, dt=dt, title="ORI_11m1", windows=windows,
    #                         show_frac=True, frac_scale=100.0)
    # plot_ori_debug_decomposed(out, counts, ORI_m111, dt=dt, title="ORI_m111", windows=windows,
    #                         show_frac=True, frac_scale=100.0)

    # win1 = (28000, 35000)
    # res1 = find_top_variance_nvs(out, ORI_11m1, dt, win1, topk=20)
    # print("Top NVs (ORI_11m1) driving var jump:", res1["top_nv_indices"])
    
    dt = 0.248
    windows = [(28000, 35000), (69000, 72000)]
    # for i in range(102):
    plot_lockin_simple(out, counts, ori_idx=ORI_m111_ORI_11m1, dt=1.0)
    # plot_per_nv_raw_DI_DQ_var(out, counts, ori_idx=None, dt=1.0, nshots=None)

    plt.show()
    sys.exit()
    res = analyze_correlated_modes(out, nv_list, K=6)
    # plot_all_pairwise(out, use="z",  mask_phase_below=0.0)   # best for coherence structure
    # or, if you want the more “physical” (brightness-weighted) signal:
    # plot_all_pairwise(out, use="s0", mask_phase_below=0.0)

    # Choose what you want to analyze:
    # x = out["z"]   # whitened (good for detecting structure independent of brightness)
    x = out["s0"]    # more “physical”, retains amplitude weighting (often better for phase-noise realism)

    Rphi, dphi = phase_increment_pearson_matrix(x, lag=1, amp_mask=None)
    plot_matrix(Rphi, title="Phase-increment Pearson corr (lag=1)")
    
    amp_thr = np.percentile(np.abs(x), 20)  # example: drop lowest 10% amplitude points
    Rphi, _ = phase_increment_pearson_matrix(x, lag=1, amp_mask=amp_thr)
    plot_matrix(Rphi, title=f"Phase-increment corr (amp_mask={amp_thr:.3g})")
    
    x = out["s0"]
    amp_thr_per_nv = np.percentile(np.abs(x), 20, axis=1)

    Rphi, dphi = phase_increment_pearson_matrix(x, lag=1, amp_mask=amp_thr_per_nv)

    # choose a score for sorting: e.g. "row strength"
    score_Rphi = np.mean(np.abs(Rphi - np.eye(Rphi.shape[0])), axis=1)

    # use sign split from the TOP MODE of C_time (so it's consistent across matrices)
    w_eval, V = eig_hermitian(out["C_time"])
    v1 = V[:, 0]

    # plot_matrix_views(
    #     Rphi,
    #     title="Phase-increment Pearson corr (lag=1)",
    #     ORI_11m1=ORI_11m1,
    #     ORI_m111=ORI_m111,
    #     sign_vec=v1,
    #     score=score_Rphi,
    #     bins=80,
    # )

    perm = perm_by_spectral_seriation(Rphi, use_abs=True)
    plot_matrix(Rphi[np.ix_(perm, perm)], title="Rphi (spectral-sorted)", show_phase=False)

    plt.show()
    # Replace this with your actual counts:
    # raise SystemExit("Edit __main__ with your counts loader (npz / dm.get_raw_data / etc.)")
