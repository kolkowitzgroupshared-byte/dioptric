import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio


def load_gif_frames(gif_path):
    """
    Load GIF into grayscale array: frames shape = T x H x W.
    """
    frames = iio.imread(gif_path)  # T x H x W x RGB or T x H x W

    frames = np.asarray(frames)

    if frames.ndim == 4:
        # RGB/RGBA -> grayscale
        frames = frames[..., :3].mean(axis=-1)

    return frames.astype(np.float32)


def load_pattern_camera_points(calib_path):
    """
    Load camera spot coordinates from saved triangle-affine mapping.
    """
    data = np.load(calib_path, allow_pickle=True)

    possible_keys = [
        "pattern_camera_points",
        "circle_camera_points",
        "slm_camera_points",
        "new_cam_pts",
        "cam_pts",
    ]

    for key in possible_keys:
        if key in data.files:
            print(f"Using camera points from key: {key}")
            return data[key]

    raise KeyError(f"No camera point key found. Available keys: {data.files}")


def integrate_spot_traces(frames, cam_pts, roi=6):
    """
    Integrate intensity around each camera spot for every GIF frame.

    frames: T x H x W
    cam_pts: N x 2, [x, y]
    returns traces: T x N
    """
    frames = np.asarray(frames).astype(np.float32)
    cam_pts = np.asarray(cam_pts).astype(np.float32)

    T, H, W = frames.shape
    N = len(cam_pts)

    traces = np.zeros((T, N), dtype=np.float32)

    for t in range(T):
        img = frames[t]

        for i, (x, y) in enumerate(cam_pts):
            x = int(round(x))
            y = int(round(y))

            x0 = max(0, x - roi)
            x1 = min(W, x + roi + 1)
            y0 = max(0, y - roi)
            y1 = min(H, y + roi + 1)

            traces[t, i] = img[y0:y1, x0:x1].sum()

    return traces


def normalize_traces(traces, baseline_frames=3):
    """
    Normalize each spot by its initial intensity.
    """
    baseline = np.median(traces[:baseline_frames], axis=0)
    baseline = np.maximum(baseline, 1.0)
    return traces / baseline


def plot_gif_crosstalk_heatmap(norm_traces):
    plt.figure(figsize=(10, 5))
    plt.imshow(
        norm_traces.T,
        aspect="auto",
        origin="lower",
        vmin=0,
        vmax=1.2,
        cmap="viridis",
    )
    plt.colorbar(label="Normalized GIF intensity")
    plt.xlabel("GIF frame")
    plt.ylabel("Spot index")
    plt.title("DMD movie: spot intensity heatmap")
    plt.tight_layout()
    plt.show()


def plot_average_intensity(norm_traces):
    mean_trace = np.mean(norm_traces, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(mean_trace, "-o")
    plt.xlabel("GIF frame")
    plt.ylabel("Mean normalized spot intensity")
    plt.title("Average spot intensity over movie")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Use these paths
# -----------------------------

gif_path = "dmdsuite/calibration/new_pattern_triangle_affine_movie.gif"
calib_path = "dmdsuite/calibration/new_pattern_from_triangle_affine.npz"

frames = load_gif_frames(gif_path)
cam_pts = load_pattern_camera_points(calib_path)

print("GIF frames shape:", frames.shape)
print("Number of camera spots:", len(cam_pts))

traces = integrate_spot_traces(frames, cam_pts, roi=6)
norm_traces = normalize_traces(traces, baseline_frames=3)

plot_gif_crosstalk_heatmap(norm_traces)
plot_average_intensity(norm_traces)

# Save analysis
np.savez(
    "dmdsuite/calibration/gif_crosstalk_analysis.npz",
    frames_shape=np.array(frames.shape),
    cam_pts=cam_pts,
    traces=traces,
    norm_traces=norm_traces,
)

print("Saved: dmdsuite/calibration/gif_crosstalk_analysis.npz")
