"""
c12_c13_movie_from_hyperfine_table_no_axes_transparent.py

Edits vs your original:
- No axes, no ticks, no grid, no 3D panes/box
- Keep the colorbar (chosen 13C colored by radius r)
- Save with transparent background (works for GIF/PNG; MP4 won’t preserve alpha)

Run:
  python c12_c13_movie_from_hyperfine_table_no_axes_transparent.py
"""

import matplotlib as mpl

mpl.use("Agg")  # comment out if you want interactive window

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    from matplotlib.animation import FFMpegWriter
except Exception:
    FFMpegWriter = None

try:
    from matplotlib.animation import PillowWriter
except Exception:
    PillowWriter = None

from matplotlib.colors import Normalize

# ===================== USER SETTINGS =====================
HYPERFINE_PATH = "analysis/nv_hyperfine_coupling/nv-2.txt"
R_CUTOFF_A = 15.0  # set None to disable (use all sites)

N_FRAMES = 200
FPS = 2
OUT_PATH = "c12_c13_sites.gif"  # ".mp4" if ffmpeg installed (MP4 won't keep alpha)

P13 = 0.011  # natural abundance ~1.1%
N_CHOSEN = None  # set int to override P13 and force exactly this many
ANG = "Å"
SEED = 1234

# Visual style
BG_ALPHA = 0.10
BG_SIZE = 5.0
CHOSEN_SIZE = 26.0
BG_DOWNSAMPLE = 1  # 1 = plot all, 2 = every other, 5 = every 5th, etc.

# Spin arrows
SHOW_SPIN_ARROWS = True
ARROW_LEN = 1.2
ARROW_LW = 1.2

# Camera rotation
ROTATE_VIEW = True
AZIM0 = 35.0
ELEV0 = 22.0
ROT_DEG_PER_FRAME = 0.8

# Colorbar / coloring of chosen points
SHOW_COLORBAR = True
CHOSEN_CMAP = "viridis"  # any matplotlib cmap name

# Save
SAVE_DPI = 300
MP4_BITRATE = 8000
# =========================================================


def read_hyperfine_table_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    def _is_int_start(s: str) -> bool:
        s = s.lstrip()
        if not s:
            return False
        t = s.split()[0]
        try:
            int(t)
            return True
        except Exception:
            return False

    try:
        data_start = next(i for i, line in enumerate(lines) if _is_int_start(line))
    except StopIteration:
        raise RuntimeError(f"Could not locate data start in hyperfine file: {path}")

    HF_COLS = [
        "index",
        "distance",
        "x",
        "y",
        "z",
        "Axx",
        "Ayy",
        "Azz",
        "Axy",
        "Axz",
        "Ayz",
    ]

    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            comment="#",
            header=None,
            names=HF_COLS,
            usecols=list(range(11)),
            skiprows=data_start,
            na_filter=False,
        )
        df = df.astype(
            {
                "index": int,
                "distance": float,
                "x": float,
                "y": float,
                "z": float,
                "Axx": float,
                "Ayy": float,
                "Azz": float,
                "Axy": float,
                "Axz": float,
                "Ayz": float,
            },
            errors="ignore",
        )
        return df
    except Exception as e:
        arr = np.loadtxt(path, comments="#", dtype=float, ndmin=2)
        if arr.shape[1] < 11:
            raise RuntimeError(
                f"Expected ≥11 columns, found {arr.shape[1]} in {path}"
            ) from e
        arr = arr[:, :11]
        df = pd.DataFrame(arr, columns=HF_COLS)
        df["index"] = df["index"].round().astype(int)
        return df


def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def make_3d_axes_invisible_but_keep_text(ax):
    """No axis box, ticks, grids, panes. Title/text still works."""
    ax.grid(False)
    ax.set_axis_off()

    # Make panes fully transparent (helps in some mpl versions)
    try:
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
    except Exception:
        pass


def main():
    df = read_hyperfine_table_safe(HYPERFINE_PATH)
    pos_all_full = df[["x", "y", "z"]].to_numpy(dtype=float)

    if R_CUTOFF_A is None:
        pos_all = pos_all_full
    else:
        r_full = np.linalg.norm(pos_all_full, axis=1)
        keep = r_full <= float(R_CUTOFF_A)
        pos_all = pos_all_full[keep]

    n_total = pos_all.shape[0]
    if n_total == 0:
        raise ValueError(f"No sites remain after applying R_CUTOFF_A={R_CUTOFF_A} Å")

    # Precompute radii for coloring
    r_all = np.linalg.norm(pos_all, axis=1)
    vmax = float(R_CUTOFF_A) if R_CUTOFF_A is not None else float(np.max(r_all))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-9))

    rng = np.random.default_rng(SEED)

    if N_CHOSEN is None:
        n_chosen = int(round(P13 * n_total))
        n_chosen = max(1, n_chosen)
    else:
        n_chosen = int(N_CHOSEN)

    if n_chosen > n_total:
        raise ValueError(f"N_CHOSEN={n_chosen} > total sites {n_total}")

    chosen_idx_frames = [
        rng.choice(n_total, size=n_chosen, replace=False) for _ in range(N_FRAMES)
    ]

    pos_bg = pos_all[:: max(1, int(BG_DOWNSAMPLE))]

    max_abs = float(np.max(np.abs(pos_all)))
    lim = max(1.0, max_abs) * 1.15

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    # Transparent backgrounds (figure + 3D axes)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

    # Background points (12C)
    ax.scatter(pos_bg[:, 0], pos_bg[:, 1], pos_bg[:, 2], s=BG_SIZE, alpha=BG_ALPHA)

    # Chosen points (13C), frame 0 (colored by r)
    idx0 = chosen_idx_frames[0]
    p0 = pos_all[idx0]
    r0 = r_all[idx0]
    scat_c13 = ax.scatter(
        p0[:, 0],
        p0[:, 1],
        p0[:, 2],
        c=r0,
        cmap=CHOSEN_CMAP,
        norm=norm,
        s=CHOSEN_SIZE,
        depthshade=True,
    )

    # NV at origin
    ax.scatter([0], [0], [0], s=110, marker="*", zorder=6)
    ax.text(0, 0, 0, "NV", fontsize=10, ha="right", va="top")

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    set_axes_equal_3d(ax)

    ax.view_init(elev=ELEV0, azim=AZIM0)

    # Title + info (kept same)
    title = ax.set_title(
        f"Carbon sites: ¹²C (I=0) faint, ¹³C (I=1/2) bright — frame 1/{N_FRAMES}"
    )
    info = ax.text2D(
        0.02,
        0.02,
        f"Total sites: {n_total}\nChosen ¹³C/frame: {n_chosen}\nP13={P13:.3g}"
        + (f"\nR_cut={R_CUTOFF_A:.1f} {ANG}" if R_CUTOFF_A is not None else "")
        + (f"\nspin arrow={ARROW_LEN:.2f} {ANG}" if SHOW_SPIN_ARROWS else ""),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, lw=0.6),
    )

    # Colorbar only (keep)
    cbar = None
    if SHOW_COLORBAR:
        cbar = fig.colorbar(scat_c13, ax=ax, shrink=0.78, pad=0.02)
        cbar.set_label(f"r ({ANG})")
        # transparent cbar background
        try:
            cbar.ax.set_facecolor((1, 1, 1, 0))
        except Exception:
            pass

    # Now remove axes/grids/panes (keep title/text + colorbar)
    make_3d_axes_invisible_but_keep_text(ax)

    # Spin arrows for 13C: random ± along z each frame
    quiv = None

    def draw_spin_arrows(pos_xyz: np.ndarray, seed_int: int):
        nonlocal quiv
        if quiv is not None:
            try:
                quiv.remove()
            except Exception:
                pass
            quiv = None
        if (not SHOW_SPIN_ARROWS) or pos_xyz.size == 0:
            return

        rrng = np.random.default_rng(seed_int)
        signs = rrng.choice([-1.0, +1.0], size=pos_xyz.shape[0])
        u = np.zeros(pos_xyz.shape[0])
        v = np.zeros(pos_xyz.shape[0])
        w = signs * ARROW_LEN
        quiv = ax.quiver(
            pos_xyz[:, 0],
            pos_xyz[:, 1],
            pos_xyz[:, 2],
            u,
            v,
            w,
            arrow_length_ratio=0.25,
            linewidth=ARROW_LW,
            alpha=0.9,
        )

    draw_spin_arrows(p0, seed_int=(SEED ^ 0xA5A5A5A5))

    def update(i: int):
        idx = chosen_idx_frames[i]
        p = pos_all[idx]
        rr = r_all[idx]

        # update positions
        scat_c13._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        # update colors (colorbar stays consistent via fixed norm)
        scat_c13.set_array(rr)

        draw_spin_arrows(p, seed_int=(SEED + i) ^ 0xA5A5A5A5)

        title.set_text(
            f"Carbon sites: ¹²C (I=0) faint, ¹³C (I=1/2) bright — frame {i+1}/{N_FRAMES}"
        )
        if ROTATE_VIEW:
            ax.view_init(elev=ELEV0, azim=AZIM0 + ROT_DEG_PER_FRAME * i)

        return tuple(x for x in (scat_c13, title, info, quiv) if x is not None)

    ani = FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        interval=int(1000 / max(1, FPS)),
        blit=False,
        repeat=False,
    )

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_bin

    savefig_kwargs = dict(
        transparent=True,
        facecolor="none",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0,
    )

    try:
        if (
            out_path.suffix.lower() == ".mp4"
            and ffmpeg_bin is not None
            and FFMpegWriter is not None
        ):
            writer = FFMpegWriter(
                fps=FPS, bitrate=MP4_BITRATE, extra_args=["-pix_fmt", "yuv420p"]
            )
            ani.save(
                str(out_path),
                writer=writer,
                dpi=SAVE_DPI,
                savefig_kwargs=savefig_kwargs,
            )
        else:
            if out_path.suffix.lower() != ".gif":
                out_path = out_path.with_suffix(".gif")
            if PillowWriter is None:
                raise RuntimeError(
                    "PillowWriter unavailable. Install pillow or ffmpeg."
                )
            ani.save(
                str(out_path),
                writer=PillowWriter(fps=FPS),
                dpi=SAVE_DPI,
                savefig_kwargs=savefig_kwargs,
            )
    except TypeError:
        # Fallback for older matplotlib without savefig_kwargs support
        if (
            out_path.suffix.lower() == ".mp4"
            and ffmpeg_bin is not None
            and FFMpegWriter is not None
        ):
            writer = FFMpegWriter(
                fps=FPS, bitrate=MP4_BITRATE, extra_args=["-pix_fmt", "yuv420p"]
            )
            ani.save(str(out_path), writer=writer, dpi=SAVE_DPI)
        else:
            if out_path.suffix.lower() != ".gif":
                out_path = out_path.with_suffix(".gif")
            if PillowWriter is None:
                raise RuntimeError(
                    "PillowWriter unavailable. Install pillow or ffmpeg."
                )
            ani.save(str(out_path), writer=PillowWriter(fps=FPS), dpi=SAVE_DPI)

    plt.close(fig)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
