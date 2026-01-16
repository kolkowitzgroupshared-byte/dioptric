# import numpy as np
# import matplotlib.pyplot as plt

# # ----------------------------
# # USER INPUTS
# # ----------------------------
# B_vec_G = np.array(
#     [-48.67047318, -32.07615947, 22.49657427], dtype=float
# )  # [100],[010],[001] basis
# observed = np.array([109, 120, 157, 168, 203, 211, 248, 263, 293, 311], dtype=float)

# fmin, fmax = 10.0, 360.0  # MHz
# gaussian_sigma_MHz = 0.2
# intensity_thresh = 1e-5

# # P1 (substitutional N, 14N) parameters (MHz)
# A_par = 114.0264
# A_perp = 81.312
# P_par = -3.9770

# gamma_e = 2.802495  # MHz/G
# gamma_n = 0.0003077  # MHz/G (14N) small


# # ----------------------------
# # SPIN OPERATORS
# # ----------------------------
# def spin_matrices_s_half():
#     Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
#     Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
#     Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
#     return Sx, Sy, Sz


# def spin_matrices_I1():
#     Ix = (1 / np.sqrt(2)) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
#     Iy = (1 / np.sqrt(2)) * np.array(
#         [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex
#     )
#     Iz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
#     return Ix, Iy, Iz


# Sx, Sy, Sz = spin_matrices_s_half()
# Ix, Iy, Iz = spin_matrices_I1()

# kron = np.kron
# Sx6, Sy6, Sz6 = kron(Sx, np.eye(3)), kron(Sy, np.eye(3)), kron(Sz, np.eye(3))
# Ix6, Iy6, Iz6 = kron(np.eye(2), Ix), kron(np.eye(2), Iy), kron(np.eye(2), Iz)


# # ----------------------------
# # GEOMETRY
# # ----------------------------
# def orthonormal_frame_from_z(z):
#     z = np.array(z, dtype=float)
#     z = z / np.linalg.norm(z)
#     a = np.array([0, 0, 1.0])
#     if abs(np.dot(a, z)) > 0.9:
#         a = np.array([0, 1.0, 0])
#     x = np.cross(a, z)
#     x /= np.linalg.norm(x)
#     y = np.cross(z, x)
#     y /= np.linalg.norm(y)
#     return np.column_stack([x, y, z])


# # Four possible JT axes (tetrahedral N–C bonds) in [100],[010],[001] basis
# jt_axes = [
#     np.array([1, 1, 1], float),
#     np.array([1, -1, -1], float),
#     np.array([-1, 1, -1], float),
#     np.array([-1, -1, 1], float),
# ]
# jt_axes = [a / np.linalg.norm(a) for a in jt_axes]
# jt_labels = ["[111]", "[1-1-1]", "[-11-1]", "[-1-11]"]


# # ----------------------------
# # HAMILTONIAN
# # ----------------------------
# def p1_hamiltonian(B_vec_G, jt_axis):
#     # Hyperfine tensor A in lab frame (principal axis = JT axis)
#     R = orthonormal_frame_from_z(jt_axis)
#     A0 = np.diag([A_perp, A_perp, A_par])
#     A = R @ A0 @ R.T

#     # Quadrupole along JT axis: P_par*(I_n^2 - I(I+1)/3)
#     n = np.array(jt_axis, dtype=float)
#     n /= np.linalg.norm(n)
#     I_n = n[0] * Ix6 + n[1] * Iy6 + n[2] * Iz6
#     I = 1
#     HQ = P_par * (I_n @ I_n - (I * (I + 1) / 3) * np.eye(6))

#     # Zeeman terms
#     Bx, By, Bz = B_vec_G
#     HZ = gamma_e * (Bx * Sx6 + By * Sy6 + Bz * Sz6) - gamma_n * (
#         Bx * Ix6 + By * Iy6 + Bz * Iz6
#     )

#     # Hyperfine S·A·I
#     S_ops = [Sx6, Sy6, Sz6]
#     I_ops = [Ix6, Iy6, Iz6]
#     Hhf = np.zeros((6, 6), dtype=complex)
#     for a in range(3):
#         for b in range(3):
#             Hhf += A[a, b] * (S_ops[a] @ I_ops[b])

#     return HZ + Hhf + HQ


# def compute_lines(B_vec_G):
#     Bhat = B_vec_G / np.linalg.norm(B_vec_G)
#     S_B = Bhat[0] * Sx6 + Bhat[1] * Sy6 + Bhat[2] * Sz6

#     # crude MW-driving operator (you can swap this if your B1 polarization differs)
#     O = Sx6 + 1j * Sy6

#     lines = []
#     for axis_id, axis in enumerate(jt_axes):
#         H = p1_hamiltonian(B_vec_G, axis)
#         evals, evecs = np.linalg.eigh(H)

#         # precompute expectation values used for labeling
#         n = axis / np.linalg.norm(axis)
#         I_n = n[0] * Ix6 + n[1] * Iy6 + n[2] * Iz6

#         ms = np.array(
#             [np.real(np.vdot(evecs[:, k], S_B @ evecs[:, k])) for k in range(6)]
#         )
#         mi = np.array(
#             [np.real(np.vdot(evecs[:, k], I_n @ evecs[:, k])) for k in range(6)]
#         )

#         for i in range(6):
#             for j in range(i + 1, 6):
#                 f = float(np.real(evals[j] - evals[i]))
#                 if not (fmin <= f <= fmax):
#                     continue
#                 amp = float(abs(np.vdot(evecs[:, i], O @ evecs[:, j])) ** 2)
#                 if amp < intensity_thresh:
#                     continue

#                 lines.append(
#                     {
#                         "f": f,
#                         "amp": amp,
#                         "axis_id": axis_id,
#                         "mi_avg": 0.5
#                         * float(mi[i] + mi[j]),  # ~ -1,0,+1 when weakly mixed
#                         "dms": float(ms[j] - ms[i]),
#                         "dmi": float(mi[j] - mi[i]),
#                     }
#                 )

#     lines.sort(key=lambda d: d["f"])
#     return lines


# lines = compute_lines(B_vec_G)

# print(f"|B| = {np.linalg.norm(B_vec_G):.3f} G")
# print("Top predicted lines (by strength):")
# for d in sorted(lines, key=lambda x: -x["amp"])[:15]:
#     print(
#         f"  {d['f']:7.2f} MHz  amp={d['amp']:.3g}  JT={jt_labels[d['axis_id']]}  <I_JT>~{d['mi_avg']:+.2f}"
#     )


# # ----------------------------
# # MATCH OBSERVED PEAKS
# # ----------------------------
# def match_observed(fobs, lines, window=12.0, topk=5):
#     cand = [d for d in lines if abs(d["f"] - fobs) <= window]
#     cand.sort(key=lambda d: (abs(d["f"] - fobs), -d["amp"]))
#     return cand[:topk]


# print("\nObserved peak assignments (candidates):")
# for fobs in observed:
#     cand = match_observed(fobs, lines, window=12.0, topk=4)
#     print(f"\nObs {fobs:.1f} MHz:")
#     if not cand:
#         print(
#             "  (no predicted lines within window) -> check B calibration or widen window"
#         )
#         continue
#     for d in cand:
#         print(
#             f"  pred {d['f']:7.2f} (Δ={d['f']-fobs:+6.2f})  amp={d['amp']:.3g}  JT={jt_labels[d['axis_id']]}  <I_JT>~{d['mi_avg']:+.2f}"
#         )

# # ----------------------------
# # PLOT
# # ----------------------------
# grid = np.linspace(fmin, fmax, 5000)
# spec = np.zeros_like(grid)
# for d in lines:
#     spec += d["amp"] * np.exp(-((grid - d["f"]) ** 2) / (2 * gaussian_sigma_MHz**2))

# plt.figure()
# for d in lines:
#     plt.vlines(d["f"], 0, d["amp"], linewidth=1)

# if spec.max() > 0:
#     plt.plot(grid, spec / spec.max())

# plt.scatter(observed, np.full_like(observed, 0.05), marker="x")
# plt.xlabel("Frequency (MHz)")
# plt.ylabel("Relative transition strength (arb.)")
# plt.title(
#     f"P1 predicted transitions (B={B_vec_G} G; |B|={np.linalg.norm(B_vec_G):.2f} G)"
# )
# plt.xlim(fmin, fmax)
# plt.tight_layout()
# # plt.savefig("p1_spectrum_labeled.png", dpi=300)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

from utils import data_manager as dm
from utils import widefield as widefield


import numpy as np
import matplotlib.pyplot as plt

from utils import data_manager as dm
from utils import widefield as widefield


def load_nv_contrast(file_id, dynamic_thresh=True):
    data = dm.get_raw_data(file_stem=file_id, load_npz=True, use_cache=True)

    nv_list = data["nv_list"]
    freqs_on = np.asarray(data["freqs"], float)  # (Nf,) GHz
    counts = np.asarray(data["counts"])  # (E, NV, runs, steps=2*Nf, reps)

    Nf = freqs_on.size
    on_idx = np.arange(0, 2 * Nf, 2)
    off_idx = np.arange(1, 2 * Nf, 2)

    E0 = counts[0]  # (NV, runs, steps, reps)
    sig_counts = E0[:, :, on_idx, :]  # (NV, runs, Nf, reps)
    ref_counts = E0[:, :, off_idx, :]  # (NV, runs, Nf, reps)

    sig_counts, ref_counts = widefield.threshold_counts(
        nv_list, sig_counts, ref_counts, dynamic_thresh=dynamic_thresh
    )

    # avoid integer overflow in downstream reductions
    sig_counts = sig_counts.astype(np.float64, copy=False)
    ref_counts = ref_counts.astype(np.float64, copy=False)

    avg_contrast, avg_contrast_ste = widefield.calc_contrast(sig_counts, ref_counts)
    return freqs_on, avg_contrast, avg_contrast_ste


# def plot_all_nvs_in_4_figures(
#     x2,
#     c2,
#     e2,
#     x4,
#     c4,
#     e4,
#     n_figs=4,
#     ncols=6,
#     use_errorbar=False,
#     height_per_row=2.8,
#     width_per_col=3.0,
#     start_index=0,
#     pad_frac=0.05,
# ):
#     """
#     Plot ALL NVs split into n_figs separate figures.
#     - No gap between subplots
#     - Autoscale EACH subplot (x from its data range; y from its own data range)
#     - X ticks only on bottom row
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     num_nvs = min(c2.shape[0], c4.shape[0])
#     indices = np.arange(start_index, num_nvs, dtype=int)
#     chunks = np.array_split(indices, n_figs)

#     for fig_i, chunk in enumerate(chunks, start=1):
#         if len(chunk) == 0:
#             continue

#         nrows = int(np.ceil(len(chunk) / ncols))
#         fig, axes = plt.subplots(
#             nrows,
#             ncols,
#             figsize=(width_per_col * ncols, height_per_row * nrows),
#             sharex=False,
#             sharey=False,  # per-subplot autoscale
#             constrained_layout=False,
#         )
#         axes = np.atleast_1d(axes).ravel()

#         for k, nv_i in enumerate(chunk):
#             ax = axes[k]

#             # plot
#             if use_errorbar:
#                 ax.errorbar(
#                     x2,
#                     c2[nv_i],
#                     yerr=e2[nv_i],
#                     marker="o",
#                     ms=2,
#                     lw=0.8,
#                     alpha=0.9,
#                     label="2 µs",
#                 )
#                 ax.errorbar(
#                     x4,
#                     c4[nv_i],
#                     yerr=e4[nv_i],
#                     marker="o",
#                     ms=2,
#                     lw=0.8,
#                     alpha=0.9,
#                     label="4 µs",
#                 )
#             else:
#                 ax.plot(x2, c2[nv_i], marker="o", ms=2, lw=0.8, alpha=0.9, label="2 µs")
#                 ax.plot(x4, c4[nv_i], marker="o", ms=2, lw=0.8, alpha=0.9, label="4 µs")

#             ax.axhline(0, ls="--", alpha=0.25)
#             ax.set_title(f"NV {nv_i}", fontsize=9, pad=2)
#             ax.grid(True, linestyle="--", alpha=0.15)

#             # -------- per-subplot autoscale (x and y) --------
#             xmin = np.nanmin([np.nanmin(x2), np.nanmin(x4)])
#             xmax = np.nanmax([np.nanmax(x2), np.nanmax(x4)])
#             ax.set_xlim(xmin, xmax)

#             yvals = np.concatenate(
#                 [np.asarray(c2[nv_i], float), np.asarray(c4[nv_i], float)]
#             )
#             yvals = yvals[np.isfinite(yvals)]
#             if yvals.size:
#                 ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
#                 if ymin == ymax:
#                     ymin -= 1.0
#                     ymax += 1.0
#                 pad = pad_frac * (ymax - ymin)
#                 ax.set_ylim(ymin - pad, ymax + pad)

#             # -------- x ticks only on bottom row --------
#             row = k // ncols
#             if row != nrows - 1:
#                 ax.set_xticklabels([])
#                 ax.tick_params(axis="x", which="both", length=0)

#         # turn off unused axes
#         for j in range(len(chunk), len(axes)):
#             axes[j].axis("off")

#         # remove gaps between subplots (and outer margins kept small)
#         fig.subplots_adjust(
#             left=0.04, right=0.995, bottom=0.05, top=0.95, wspace=0.0, hspace=0.0
#         )

#         # one legend per figure
#         handles, labels = axes[0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc="upper right", frameon=True)

#         fig.supxlabel("RF frequency (MHz)")
#         fig.supylabel("Contrast")
#         fig.suptitle(
#             f"DEER contrast (2 µs vs 4 µs) — Figure {fig_i}/{n_figs} (NVs {chunk[0]}–{chunk[-1]})",
#             y=0.995,
#         )

#         plt.show()


def plot_all_nvs_in_4_figures(
    x2,
    c2,
    e2,
    x4,
    c4,
    e4,
    n_figs=4,
    ncols=6,
    use_errorbar=False,
    height_per_row=2.8,
    width_per_col=4.0,
    start_index=0,
    pad_frac=0.05,
    save_dir=None,  # e.g. "deer_4figs"
    save_prefix="deer_2us_vs_4us",
    dpi=200,
    save_png=True,
    save_pdf=False,
):
    """
    Plot ALL NVs split into n_figs separate figures.
    - No gap between subplots
    - Autoscale EACH subplot
    - X ticks only on bottom row
    - Optionally save each figure
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    num_nvs = min(c2.shape[0], c4.shape[0])
    indices = np.arange(start_index, num_nvs, dtype=int)
    chunks = np.array_split(indices, n_figs)

    saved_paths = []

    for fig_i, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue

        nrows = int(np.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(width_per_col * ncols, height_per_row * nrows),
            sharex=False,
            sharey=False,
            constrained_layout=False,
        )
        axes = np.atleast_1d(axes).ravel()

        for k, nv_i in enumerate(chunk):
            ax = axes[k]

            if use_errorbar:
                ax.errorbar(
                    x2,
                    c2[nv_i],
                    yerr=e2[nv_i],
                    marker="o",
                    ms=2,
                    lw=0.8,
                    alpha=0.9,
                    label="2 µs",
                )
                ax.errorbar(
                    x4,
                    c4[nv_i],
                    yerr=e4[nv_i],
                    marker="o",
                    ms=2,
                    lw=0.8,
                    alpha=0.9,
                    label="4 µs",
                )
            else:
                ax.plot(x2, c2[nv_i], marker="o", ms=2, lw=0.8, alpha=0.9, label="2 µs")
                ax.plot(x4, c4[nv_i], marker="o", ms=2, lw=0.8, alpha=0.9, label="4 µs")

            ax.axhline(0, ls="--", alpha=0.25)
            ax.set_title(f"NV {nv_i}", fontsize=9, pad=2)
            ax.grid(True, linestyle="--", alpha=0.15)

            # per-subplot autoscale
            xmin = np.nanmin([np.nanmin(x2), np.nanmin(x4)])
            xmax = np.nanmax([np.nanmax(x2), np.nanmax(x4)])
            ax.set_xlim(xmin, xmax)

            yvals = np.concatenate(
                [np.asarray(c2[nv_i], float), np.asarray(c4[nv_i], float)]
            )
            yvals = yvals[np.isfinite(yvals)]
            if yvals.size:
                ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                pad = pad_frac * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

            # x ticks only on bottom row
            row = k // ncols
            if row != nrows - 1:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", length=0)

        for j in range(len(chunk), len(axes)):
            axes[j].axis("off")

        fig.subplots_adjust(
            left=0.04, right=0.995, bottom=0.05, top=0.95, wspace=0.0, hspace=0.0
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=True)

        fig.supxlabel("RF frequency (MHz)")
        fig.supylabel("Contrast")
        fig.suptitle(
            f"DEER contrast (2 µs vs 4 µs) — Figure {fig_i}/{n_figs} (NVs {chunk[0]}–{chunk[-1]})",
            y=0.995,
        )

        # -------- SAVE --------
        if save_dir is not None:
            base = f"{save_prefix}_fig{fig_i:02d}_NV{chunk[0]:03d}-{chunk[-1]:03d}"
            if save_png:
                png_path = os.path.join(save_dir, base + ".png")
                fig.savefig(png_path, dpi=dpi)
                saved_paths.append(png_path)
            if save_pdf:
                pdf_path = os.path.join(save_dir, base + ".pdf")
                fig.savefig(pdf_path)
                saved_paths.append(pdf_path)

        plt.show(block=True)

    if save_dir is not None:
        print("Saved:")
        for p in saved_paths:
            print("  ", p)
    return saved_paths


# Example:


# Example call:


def plot_one_nv_overlay(x2, c2, e2, x4, c4, e4, nv_i, use_errorbar=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))  # taller

    if use_errorbar:
        ax.errorbar(x2, c2[nv_i], yerr=e2[nv_i], marker="o", ms=3, lw=1, label="2 µs")
        ax.errorbar(x4, c4[nv_i], yerr=e4[nv_i], marker="o", ms=3, lw=1, label="4 µs")
    else:
        ax.plot(x2, c2[nv_i], marker="o", ms=3, lw=1, label="2 µs")
        ax.plot(x4, c4[nv_i], marker="o", ms=3, lw=1, label="4 µs")

    ax.axhline(0, ls="--", alpha=0.3)
    ax.set_title(f"NV {nv_i} — DEER contrast (2 µs vs 4 µs)")
    ax.set_xlabel("RF frequency (MHz)")
    ax.set_ylabel("Contrast")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show(block=True)


# -------------------------
# Set your two datasets here
# -------------------------
file_2us = [
    "2026_01_11-04_19_03-johnson-nv0_2025_10_21",
    "2026_01_11-12_50_25-johnson-nv0_2025_10_21",
]  # 2000ns DEER
file_4us = ["2026_01_12-11_42_09-johnson-nv0_2025_10_21"]  # 4000ns DEER

f2, c2, e2 = load_nv_contrast(file_2us)
f4, c4, e4 = load_nv_contrast(file_4us)

x2 = f2 * 1000.0  # MHz
x4 = f4 * 1000.0  # MHz

## Example calls:
plot_all_nvs_in_4_figures(
    x2,
    c2,
    e2,
    x4,
    c4,
    e4,
    n_figs=4,
    ncols=6,
    use_errorbar=False,
    save_dir="deer_4figs",
    save_prefix="johnson_deer",
    dpi=250,
)
# plot_all_nvs_in_4_figures(x2, c2, e2, x4, c4, e4, n_figs=4, ncols=6, use_errorbar=False)

# for nv_i in range(c2.shape[0]):
#     plot_one_nv_overlay(x2, c2, e2, x4, c4, e4, nv_i=nv_i, use_errorbar=False)
# from pathlib import Path
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
# import numpy as np


# def save_all_nv_plots(
#     x2,
#     c2,
#     e2,
#     x4,
#     c4,
#     e4,
#     indices=None,
#     use_errorbar=False,
#     dpi=200,
#     make_pdf=True,
# ):
#     num_nvs = min(c2.shape[0], c4.shape[0])
#     if indices is None:
#         indices = range(num_nvs)

#     xmin = min(np.nanmin(x2), np.nanmin(x4))
#     xmax = max(np.nanmax(x2), np.nanmax(x4))

#     for nv_i in indices:
#         fig, ax = plt.subplots(figsize=(7.5, 4.8))  # taller

#         if use_errorbar:
#             ax.errorbar(
#                 x2, c2[nv_i], yerr=e2[nv_i], marker="o", ms=3, lw=1, label="2 µs"
#             )
#             ax.errorbar(
#                 x4, c4[nv_i], yerr=e4[nv_i], marker="o", ms=3, lw=1, label="4 µs"
#             )
#         else:
#             ax.plot(x2, c2[nv_i], marker="o", ms=3, lw=1, label="2 µs")
#             ax.plot(x4, c4[nv_i], marker="o", ms=3, lw=1, label="4 µs")

#         ax.set_xlim(xmin, xmax)
#         ax.axhline(0, ls="--", alpha=0.3)
#         ax.set_title(f"NV {nv_i} — DEER contrast (2 µs vs 4 µs)")
#         ax.set_xlabel("RF frequency (MHz)")
#         ax.set_ylabel("Contrast")
#         ax.grid(True, linestyle="--", alpha=0.25)
#         ax.legend()

#         fig.tight_layout()
#         timestamp = dm.get_time_stamp()
#         file_path = dm.get_file_path(__file__, timestamp, "deer_2us_vs_4us.png")
#         dm.save_figure(fig, file_path)
#         # if pdf is not None:
#         #     pdf.savefig(fig)
#         plt.close(fig)


# # example:
# save_all_nv_plots(x2, c2, e2, x4, c4, e4, use_errorbar=False, make_pdf=True)
