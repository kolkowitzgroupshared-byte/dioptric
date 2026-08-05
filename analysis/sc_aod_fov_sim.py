"""
FOV simulator for an AOD + 4f relay + infinity-corrected objective + camera.

It computes three *upper bounds* on the sample-plane field you can use:
  (A) AOD-limited scan range (from bandwidth Δf)
  (B) Objective corrected field (from FN)
  (C) Camera sensor FOV (from sensor size / magnification)

Final usable FOV is typically the MIN of these (and in practice can be smaller
due to clipping/vignetting, imperfect 4f, AOD efficiency roll-off, etc.).
"""

from dataclasses import dataclass
import math

# plotting
import numpy as np
import matplotlib.pyplot as plt
from utils import kplotlib as kpl

kpl.init_kplotlib()


@dataclass
class Objective:
    M_nom: float = 100.0  # nominal magnification printed on objective
    NA: float = 0.95
    FN_mm: float = 26.5  # field number (mm) at intermediate image plane
    f_tube_nom_mm: float = (
        180.0  # Olympus infinity systems are typically 180 mm tube lens
    )

    def f_obj_mm(self) -> float:
        # objective focal length (design): f_obj = f_tube_nom / M_nom
        return self.f_tube_nom_mm / self.M_nom

    def magnification_eff(self, f_tube_actual_mm: float) -> float:
        # if you use a non-design tube lens, magnification scales ~ linearly
        return self.M_nom * (f_tube_actual_mm / self.f_tube_nom_mm)

    def fn_limited_sample_fov_um(self, f_tube_actual_mm: float) -> float:
        # FN is a diameter (mm) at intermediate image plane; sample diameter = FN / M_eff
        M_eff = self.magnification_eff(f_tube_actual_mm)
        return (self.FN_mm / M_eff) * 1000.0  # mm -> µm


@dataclass
class Camera:
    nx: int = 512
    ny: int = 512
    pixel_um: float = 16.0

    def sensor_size_mm(self):
        sx = self.nx * self.pixel_um / 1000.0
        sy = self.ny * self.pixel_um / 1000.0
        return sx, sy

    def sample_fov_um(self, M_eff: float):
        sx_mm, sy_mm = self.sensor_size_mm()
        return (sx_mm / M_eff) * 1000.0, (sy_mm / M_eff) * 1000.0  # mm -> µm


@dataclass
class AOD4f:
    f1_m: float = 0.30  # first lens focal length (meters)
    f2_m: float = 0.50  # second lens focal length (meters)
    delta_f_Hz: float = 45e6  # usable bandwidth (peak-to-peak) in Hz
    wavelength_m: float = 532e-9
    v_acoustic_m_s: float = 617.0  # <-- replace with datasheet value if known

    def theta_pp_rad(self) -> float:
        # peak-to-peak deflection at AOD output (approx): Δθ = (λ / v) Δf
        return (self.wavelength_m / self.v_acoustic_m_s) * self.delta_f_Hz

    def theta_obj_pp_rad(self) -> float:
        # 4f telescope scales angles by f1/f2
        return self.theta_pp_rad() * (self.f1_m / self.f2_m)

    def sample_scan_fov_um(self, f_obj_mm: float) -> float:
        # sample scan (peak-to-peak) ≈ f_obj * Δθ_obj
        f_obj_m = f_obj_mm / 1000.0
        return (f_obj_m * self.theta_obj_pp_rad()) * 1e6  # meters -> µm


def summarize_fov(
    obj: Objective,
    cam: Camera,
    aod: AOD4f,
    f_tube_actual_mm: float = 180.0,
):
    f_obj_mm = obj.f_obj_mm()
    M_eff = obj.magnification_eff(f_tube_actual_mm)

    # AOD-limited (1D, peak-to-peak)
    fov_aod_um = aod.sample_scan_fov_um(f_obj_mm)

    # Objective FN-limited (diameter at sample)
    fov_fn_um = obj.fn_limited_sample_fov_um(f_tube_actual_mm)

    # Camera-limited
    fov_cam_x_um, fov_cam_y_um = cam.sample_fov_um(M_eff)

    # "Usable" square-ish FOV bounds
    usable_x = min(fov_aod_um, fov_cam_x_um, fov_fn_um)
    usable_y = min(fov_aod_um, fov_cam_y_um, fov_fn_um)

    out = {
        "objective_f_obj_mm": f_obj_mm,
        "M_eff": M_eff,
        "AOD_scan_pp_um (1D)": fov_aod_um,
        "Objective_FN_limit_um (diameter)": fov_fn_um,
        "Camera_limit_um (x,y)": (fov_cam_x_um, fov_cam_y_um),
        "Predicted_usable_um (x,y)": (usable_x, usable_y),
    }
    return out


def plot_fov_vs_f2_multi_bandwidth(
    obj: Objective,
    cam: Camera,
    base_aod: AOD4f,
    f_tube_actual_mm: float = 180.0,
    f2_min_m: float = 0.25,
    f2_max_m: float = 0.60,
    n: int = 250,
    bandwidths_MHz=(10, 20, 30, 45, 60),
    bold_MHz: float = 45.0,
    mark_f2_m=(0.30, 0.40, 0.50),
):
    """
    Curves: predicted AOD scan (p-p, 1D) vs f2 for multiple AOD bandwidths.
    Keeps bold_MHz (default 45 MHz) as a thicker (bold) line.
    Also overlays FN + camera limits.
    """
    f2_vals = np.linspace(f2_min_m, f2_max_m, n)

    # limits (do not depend on f2 or bandwidth)
    f_obj_mm = obj.f_obj_mm()
    M_eff = obj.magnification_eff(f_tube_actual_mm)
    fn_limit_um = obj.fn_limited_sample_fov_um(f_tube_actual_mm)
    cam_x_um, cam_y_um = cam.sample_fov_um(M_eff)

    # plot
    plt.figure(figsize=(8.5, 5.2))

    for bw_MHz in bandwidths_MHz:
        bw_Hz = float(bw_MHz) * 1e6

        scan = np.zeros_like(f2_vals)
        for i, f2 in enumerate(f2_vals):
            aod = AOD4f(
                f1_m=base_aod.f1_m,
                f2_m=float(f2),
                delta_f_Hz=bw_Hz,
                wavelength_m=base_aod.wavelength_m,
                v_acoustic_m_s=base_aod.v_acoustic_m_s,
            )
            scan[i] = aod.sample_scan_fov_um(f_obj_mm)

        lw = 3.0 if abs(bw_MHz - bold_MHz) < 1e-9 else 1.4
        plt.plot(f2_vals * 100.0, scan, linewidth=lw, label=f"Δf = {bw_MHz:g} MHz")

    # marked f2 points for the bold curve
    mark_f2_m = np.array(mark_f2_m, dtype=float)
    mark_scan = []
    for f2 in mark_f2_m:
        aod = AOD4f(
            f1_m=base_aod.f1_m,
            f2_m=float(f2),
            delta_f_Hz=float(bold_MHz) * 1e6,
            wavelength_m=base_aod.wavelength_m,
            v_acoustic_m_s=base_aod.v_acoustic_m_s,
        )
        mark_scan.append(aod.sample_scan_fov_um(f_obj_mm))
    mark_scan = np.array(mark_scan)
    plt.scatter(mark_f2_m * 100.0, mark_scan, label=f"Marked f2 (Δf={bold_MHz:g} MHz)")

    # overlay limits
    plt.axhline(
        fn_limit_um,
        linestyle="--",
        label=f"Objective FN limit (~{fn_limit_um:.0f} µm diameter)",
    )
    # plt.axhline(cam_x_um, linestyle="--", label=f"Camera limit X (~{cam_x_um:.0f} µm)")
    # plt.axhline(cam_y_um, linestyle=":",  label=f"Camera limit Y (~{cam_y_um:.0f} µm)")

    plt.xlabel("Second lens focal length f2 (cm) with f1=30cm")
    plt.ylabel("Predicted sample scan range (µm)")
    plt.title("Predicted AOD scan range vs 4f second lens (multiple AOD bandwidths)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Your objective ---
    obj = Objective(M_nom=100.0, NA=0.95, FN_mm=26.5, f_tube_nom_mm=180.0)

    # --- Your camera (edit to match your EMCCD) ---
    cam = Camera(nx=512, ny=512, pixel_um=16.0)

    # --- Your AOD + 4f parameters ---
    base = AOD4f(
        f1_m=0.30,
        f2_m=0.50,
        delta_f_Hz=45e6,
        wavelength_m=532e-9,  # <-- set your actual wavelength
        v_acoustic_m_s=617.0,  # <-- set from AOD datasheet
    )

    # Tube lens focal length actually used in your infinity system:
    f_tube_actual_mm = 180.0

    # NEW: curves for multiple bandwidths, keeping 45 MHz bold
    plot_fov_vs_f2_multi_bandwidth(
        obj=obj,
        cam=cam,
        base_aod=base,
        f_tube_actual_mm=f_tube_actual_mm,
        f2_min_m=0.20,
        f2_max_m=0.60,
        n=250,
        bandwidths_MHz=(10, 20, 30, 45, 60, 80, 100),
        bold_MHz=45.0,
        mark_f2_m=(0.30, 0.40, 0.50),
    )

    # Your original prints (unchanged)
    for f2_m in [0.50, 0.40, 0.30]:
        aod = base
        aod.f2_m = f2_m
        res = summarize_fov(obj, cam, aod, f_tube_actual_mm=f_tube_actual_mm)

        print(
            "\n=== 4f with f1=%.0f cm, f2=%.0f cm ==="
            % (aod.f1_m * 100, aod.f2_m * 100)
        )
        print(
            "Objective f_obj = %.3f mm,  M_eff = %.2f×"
            % (res["objective_f_obj_mm"], res["M_eff"])
        )
        print("AOD scan (peak-to-peak, 1D)     : %.1f µm" % res["AOD_scan_pp_um (1D)"])
        print(
            "Objective FN limit (diameter)    : %.1f µm"
            % (res["Objective_FN_limit_um (diameter)"])
        )
        cx, cy = res["Camera_limit_um (x,y)"]
        print("Camera limit (x, y)              : (%.1f, %.1f) µm" % (cx, cy))
        ux, uy = res["Predicted_usable_um (x,y)"]
        print("Predicted usable (x, y)          : (%.1f, %.1f) µm" % (ux, uy))

    print(
        "\nNOTE: If you measure only ~34×34 µm, you are likely limited by clipping/vignetting,"
    )
    print(
        "AOD efficiency roll-off (effective Δf smaller), or relay mis-conjugation—not FN."
    )
    kpl.show(block=True)
