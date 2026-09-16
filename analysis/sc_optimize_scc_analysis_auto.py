# -*- coding: utf-8 -*-
"""
Automatic SCC parameter-sweep analysis with per-NV saved-amplitude plots.

What is new relative to the previous version
--------------------------------------------
1. Still determines automatically whether the experiment varied
   - SCC AOD amplitude multiplier, or
   - SCC pulse duration.

2. Still resolves SCC defaults from the active Dioptric config:
   - SCC physical laser
   - default SCC duration
   - SCC AOD pulse / waveform
   - SCC AOD base waveform sample

3. NEW:
   - plots the fitted optimum parameter for each NV versus NV index
   - tries to extract SCC-related amplitude information already saved in nv_sig
   - plots those per-NV saved amplitudes / weights if found
   - for amplitude sweeps, compares saved SCC amplitude vs fitted optimum

This is written to be robust against slightly different nv_sig schemas by
trying several likely attribute/key names.

@author: Saroj Chand
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import matplotlib.pyplot as plt
import numpy as np

from utils import common
from utils import data_manager as dm
from utils import kplotlib as kpl
from utils import widefield
from utils.constants import VirtualLaserKey


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PeakEstimate:
    x_opt: float
    y_opt: float
    x_unc: float
    method: str
    status: str
    raw_x_max: float
    raw_y_max: float
    fit_x: np.ndarray | None = None
    fit_y: np.ndarray | None = None


@dataclass
class NVSeries:
    name: str
    label: str
    values: np.ndarray
    source: str


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def _is_mapping_like(obj):
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def _has_get_method(obj):
    return hasattr(obj, "get")


def _scalarize_numeric(value):
    """
    Convert a candidate value into a float when possible.
    Otherwise return None.
    """
    if value is None:
        return None

    if isinstance(value, numbers.Number):
        value = float(value)
        if np.isfinite(value):
            return value
        return None

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = float(value)
            return value if np.isfinite(value) else None
        return None

    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], numbers.Number):
            value = float(value[0])
            return value if np.isfinite(value) else None
        return None

    return None


def _get_attr_or_key(obj, name, default=None):
    """
    Try attribute first, then mapping-style access.
    """
    if hasattr(obj, name):
        return getattr(obj, name)

    if _has_get_method(obj):
        try:
            return obj.get(name, default)
        except Exception:
            pass

    if _is_mapping_like(obj):
        try:
            return obj[name]
        except Exception:
            pass

    return default


def _candidate_virtual_keys():
    """
    Candidate keys to retrieve SCC values from dict-like containers.

    Supports enums, enum names/values, and common string variants.
    """
    keys = []

    # Exact enum object.
    keys.append(VirtualLaserKey.SCC)

    # Safer string forms.
    for maybe in [
        getattr(VirtualLaserKey.SCC, "name", None),
        getattr(VirtualLaserKey.SCC, "value", None),
        str(VirtualLaserKey.SCC),
        "SCC",
        "scc",
        "VirtualLaserKey.SCC",
    ]:
        if maybe is not None:
            keys.append(maybe)

    # Remove duplicates while preserving order.
    deduped = []
    seen = set()
    for key in keys:
        marker = repr(key)
        if marker not in seen:
            deduped.append(key)
            seen.add(marker)
    return deduped


def _extract_from_container(container, preferred_keys):
    """
    Extract a numeric scalar from a mapping-like container using multiple keys.
    """
    if container is None:
        return None

    if _has_get_method(container):
        for key in preferred_keys:
            try:
                value = container.get(key, None)
                value = _scalarize_numeric(value)
                if value is not None:
                    return value
            except Exception:
                pass

    if _is_mapping_like(container):
        for key in preferred_keys:
            try:
                value = container[key]
                value = _scalarize_numeric(value)
                if value is not None:
                    return value
            except Exception:
                pass

    return None


# =============================================================================
# CONFIG RESOLUTION
# =============================================================================

def get_scc_config_info():
    """
    Resolve SCC defaults from the active Dioptric configuration.
    """
    config_module = common.get_config_module()
    config = config_module.config
    opx_config = config_module.opx_config

    scc_virtual = config["Optics"]["VirtualLasers"][VirtualLaserKey.SCC]
    physical_laser_name = scc_virtual["physical_name"]
    default_duration_ns = float(scc_virtual["duration"])

    possible_elements = [
        f"ao_{physical_laser_name}_x",
        f"ao_{physical_laser_name}_y",
        f"ao_{physical_laser_name}_x_2",
    ]

    selected_element = None
    selected_operation = None
    pulse_name = None

    for element_name in possible_elements:
        element = opx_config["elements"].get(element_name)
        if element is None:
            continue

        operations = element.get("operations", {})

        if "aod_cw-scc" in operations:
            selected_element = element_name
            selected_operation = "aod_cw-scc"
            pulse_name = operations["aod_cw-scc"]
            break

    if pulse_name is None:
        for element_name in possible_elements:
            element = opx_config["elements"].get(element_name)
            if element is None:
                continue

            operations = element.get("operations", {})

            if "aod_cw" in operations:
                candidate = operations["aod_cw"]
                if "scc" in str(candidate).lower():
                    selected_element = element_name
                    selected_operation = "aod_cw"
                    pulse_name = candidate
                    break

    if pulse_name is None:
        raise KeyError(
            "Could not resolve the SCC AOD pulse from OPX config."
        )

    pulse = opx_config["pulses"][pulse_name]
    waveform_name = pulse["waveforms"]["single"]
    waveform = opx_config["waveforms"][waveform_name]

    if waveform.get("type") != "constant":
        raise ValueError(
            f"SCC waveform {waveform_name!r} is not constant."
        )

    base_sample = float(waveform["sample"])

    return {
        "physical_laser_name": physical_laser_name,
        "default_duration_ns": default_duration_ns,
        "aod_element": selected_element,
        "aod_operation": selected_operation,
        "aod_pulse_name": pulse_name,
        "aod_waveform_name": waveform_name,
        "aod_base_sample": base_sample,
    }


# =============================================================================
# EXPERIMENT METADATA / PARAMETER DETECTION
# =============================================================================

def get_scan_values(data):
    if "step_vals" in data:
        vals = np.asarray(data["step_vals"], dtype=float).ravel()
        source_key = "step_vals"
    elif "taus" in data:
        vals = np.asarray(data["taus"], dtype=float).ravel()
        source_key = "taus"
    else:
        raise KeyError(
            'Experiment data contains neither "step_vals" nor legacy "taus".'
        )

    if vals.size == 0 or not np.any(np.isfinite(vals)):
        raise ValueError("Experiment sweep axis is empty or non-finite.")

    return vals, source_key


def _normalize_scan_type(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    amp_tokens = {
        "amp",
        "amplitude",
        "scc_amp",
        "scc_amplitude",
        "aod_amplitude",
        "aod amplitude",
    }

    duration_tokens = {
        "duration",
        "dur",
        "scc_duration",
        "pulse_duration",
        "pulse duration",
    }

    if text in amp_tokens:
        return "amplitude"

    if text in duration_tokens:
        return "duration"

    return None


def determine_varied_parameter(data, step_vals):
    metadata_keys = [
        "scan_type",
        "sweep_parameter",
        "varied_parameter",
        "parameter_name",
    ]

    for key in metadata_keys:
        if key in data:
            scan_type = _normalize_scan_type(data[key])
            if scan_type is not None:
                return scan_type, f'explicit metadata data["{key}"]'

    if "duration_or_amp" in data:
        scan_type = "duration" if bool(data["duration_or_amp"]) else "amplitude"
        return scan_type, 'metadata data["duration_or_amp"]'

    for key, value in data.items():
        if isinstance(value, str):
            lower = value.lower()
            if "optimize_scc-amp" in lower or "optimize_scc_amp" in lower:
                return "amplitude", f'sequence/routine string data["{key}"]'
            if "optimize_scc-duration" in lower or "optimize_scc_duration" in lower:
                return "duration", f'sequence/routine string data["{key}"]'

    for key in ["step_units", "sweep_units", "parameter_units"]:
        if key not in data:
            continue

        units = str(data[key]).strip().lower()

        if any(
            token in units
            for token in ["amplitude", "aod", "relative", "multiplier", "factor"]
        ):
            return "amplitude", f'units metadata data["{key}"]'

        if units in {"ns", "nanosecond", "nanoseconds"}:
            return "duration", f'units metadata data["{key}"]'

    finite = np.asarray(step_vals, dtype=float)
    finite = finite[np.isfinite(finite)]

    xmin = float(np.min(finite))
    xmax = float(np.max(finite))
    xmedian = float(np.median(finite))

    if xmax <= 3.0:
        return "amplitude", f"legacy numerical inference ({xmin:g}..{xmax:g}, O(1))"

    if xmin >= 4.0 or xmedian >= 4.0:
        return "duration", f"legacy numerical inference ({xmin:g}..{xmax:g} ns-scale)"

    raise ValueError(
        "Could not determine whether the experiment varied SCC amplitude "
        "or duration."
    )


# =============================================================================
# FIT HELPERS
# =============================================================================

def safe_sigma(sigma):
    sigma = np.asarray(sigma, dtype=float).copy()
    good = np.isfinite(sigma) & (sigma > 0)

    if np.any(good):
        fallback = float(np.nanmedian(sigma[good]))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 1.0

    sigma[~good] = fallback

    positive = sigma[np.isfinite(sigma) & (sigma > 0)]
    if positive.size:
        floor = max(0.25 * np.nanpercentile(positive, 10), np.finfo(float).eps)
        sigma = np.maximum(sigma, floor)

    return sigma


def quadratic_value(coeff, x, x0):
    a, b, c = coeff
    dx = np.asarray(x, dtype=float) - x0
    return a * dx**2 + b * dx + c


def estimate_peak(
    x,
    y,
    yerr,
    local_fit_points,
    max_vertex_uncertainty_fraction,
):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    yerr = np.asarray(yerr, dtype=float).ravel()

    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) == 0:
        return PeakEstimate(
            np.nan, np.nan, np.nan,
            "none", "no_valid_points",
            np.nan, np.nan,
        )

    xx = x[valid]
    yy = y[valid]
    ss = safe_sigma(yerr[valid])

    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    ss = ss[order]

    i_max = int(np.nanargmax(yy))
    raw_x = float(xx[i_max])
    raw_y = float(yy[i_max])

    if xx.size < 3:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "too_few_points",
            raw_x, raw_y,
        )

    nfit = int(np.clip(local_fit_points, 3, xx.size))
    local_inds = np.argsort(np.abs(xx - raw_x))[:nfit]
    local_inds = np.sort(local_inds)

    xf = xx[local_inds]
    yf = yy[local_inds]
    sf = ss[local_inds]

    if np.unique(xf).size < 3:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "insufficient_distinct_x",
            raw_x, raw_y,
        )

    x0 = raw_x
    dx = xf - x0

    design = np.column_stack([dx**2, dx, np.ones_like(dx)])
    weighted_design = design / sf[:, None]
    weighted_y = yf / sf

    try:
        coeff, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    except np.linalg.LinAlgError:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "lstsq_failed",
            raw_x, raw_y,
        )

    if rank < 3:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "rank_deficient",
            raw_x, raw_y,
        )

    a, b, _ = coeff

    if not np.isfinite(a) or a >= 0:
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "non_concave_fit",
            raw_x, raw_y,
        )

    vertex = x0 - b / (2.0 * a)
    local_lo = float(np.min(xf))
    local_hi = float(np.max(xf))

    if not (local_lo <= vertex <= local_hi):
        return PeakEstimate(
            raw_x, raw_y, np.nan,
            "measured_max", "vertex_outside_local_window",
            raw_x, raw_y,
        )

    try:
        covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
        grad = np.array([b / (2.0 * a**2), -1.0 / (2.0 * a), 0.0])
        variance_vertex = float(grad @ covariance @ grad)
        x_unc = np.sqrt(max(variance_vertex, 0.0))
    except Exception:
        x_unc = np.nan

    scan_span = float(np.max(xx) - np.min(xx))
    if np.isfinite(x_unc) and scan_span > 0 and x_unc > max_vertex_uncertainty_fraction * scan_span:
        return PeakEstimate(
            raw_x, raw_y, x_unc,
            "measured_max", "vertex_poorly_constrained",
            raw_x, raw_y,
        )

    y_vertex = float(quadratic_value(coeff, np.array([vertex]), x0)[0])
    fit_x = np.linspace(local_lo, local_hi, 300)
    fit_y = quadratic_value(coeff, fit_x, x0)

    return PeakEstimate(
        float(vertex),
        y_vertex,
        float(x_unc) if np.isfinite(x_unc) else np.nan,
        "local_weighted_quadratic",
        "ok",
        raw_x,
        raw_y,
        fit_x,
        fit_y,
    )


# =============================================================================
# NV-SIG AMPLITUDE / WEIGHT EXTRACTION
# =============================================================================

def extract_saved_scc_amplitude_series(nv_list):
    """
    Try to extract per-NV SCC amplitude values already stored in nv_sig.

    We try, in order:
    1. direct scalar attrs/keys such as scc_amp, scc_amplitude, ...
    2. dict-like attrs such as pulse_amps[...] keyed by VirtualLaserKey.SCC
    3. other common dict-like names such as amplitudes / laser_amps
    """
    direct_names = [
        "scc_amp",
        "scc_amps",
        "scc_amplitude",
        "scc_amplitudes",
        "scc_aod_amp",
        "scc_aod_amplitude",
        "amp_scc",
        "amplitude_scc",
        "pulse_amp_scc",
        "pulse_amplitude_scc",
    ]

    container_names = [
        "pulse_amps",
        "pulse_amplitudes",
        "laser_amps",
        "laser_amplitudes",
        "amps",
        "amplitudes",
        "amp_factors",
        "pulse_amp_factors",
        "pulse_factors",
    ]

    preferred_keys = _candidate_virtual_keys()

    # Try direct names first.
    for name in direct_names:
        values = []
        ok_count = 0

        for nv in nv_list:
            value = _get_attr_or_key(nv, name, None)
            value = _scalarize_numeric(value)
            values.append(np.nan if value is None else value)
            if value is not None:
                ok_count += 1

        if ok_count >= max(3, int(0.8 * len(nv_list))):
            return NVSeries(
                name=name,
                label=f"Saved SCC amplitude ({name})",
                values=np.asarray(values, dtype=float),
                source=f"direct nv_sig field '{name}'",
            )

    # Then try container-like names.
    for container_name in container_names:
        values = []
        ok_count = 0

        for nv in nv_list:
            container = _get_attr_or_key(nv, container_name, None)
            value = _extract_from_container(container, preferred_keys)
            values.append(np.nan if value is None else value)
            if value is not None:
                ok_count += 1

        if ok_count >= max(3, int(0.8 * len(nv_list))):
            return NVSeries(
                name=container_name,
                label=f"Saved SCC amplitude ({container_name}[SCC])",
                values=np.asarray(values, dtype=float),
                source=f"nv_sig container '{container_name}' keyed by SCC",
            )

    return None


def extract_optional_scalar_nv_series(nv_list, field_name, label=None):
    """
    Extract a scalar numeric field from each nv_sig, if present in most NVs.
    """
    values = []
    ok_count = 0

    for nv in nv_list:
        value = _get_attr_or_key(nv, field_name, None)
        value = _scalarize_numeric(value)
        values.append(np.nan if value is None else value)
        if value is not None:
            ok_count += 1

    if ok_count >= max(3, int(0.8 * len(nv_list))):
        return NVSeries(
            name=field_name,
            label=label or field_name,
            values=np.asarray(values, dtype=float),
            source=f"direct nv_sig field '{field_name}'",
        )

    return None


def gather_nv_saved_series(nv_list):
    """
    Collect per-NV amplitude/weight series that are useful to inspect.

    Always tries to get:
    - saved SCC amplitude in nv_sig
    Optionally also tries:
    - slm_amplitude_weight
    - slm_mean_norm_intensity_weight_clipped
    - slm_mean_norm_intensity_weight
    """
    series_list = []

    scc_series = extract_saved_scc_amplitude_series(nv_list)
    if scc_series is not None:
        series_list.append(scc_series)

    for field_name, label in [
        ("slm_amplitude_weight", "SLM amplitude weight"),
        ("slm_mean_norm_intensity_weight_clipped", "SLM intensity weight (clipped)"),
        ("slm_mean_norm_intensity_weight", "SLM intensity weight"),
    ]:
        series = extract_optional_scalar_nv_series(nv_list, field_name, label)
        if series is not None:
            # avoid accidental duplicates by name
            already = any(series.name == s.name for s in series_list)
            if not already:
                series_list.append(series)

    return series_list



def set_nv_scc_parameter(nv_sig, scan_type, value):
    """
    Write the fitted SCC parameter back into one NVSig.

    amplitude:
        nv_sig.pulse_amps[VirtualLaserKey.SCC] = fitted multiplier

    duration:
        nv_sig.pulse_durations[VirtualLaserKey.SCC] = fitted duration in ns

    Supports both normal NVSig objects and mapping-like representations.
    """
    if value is None or not np.isfinite(value):
        return False

    value = float(value)

    if scan_type == "amplitude":
        container_name = "pulse_amps"
    elif scan_type == "duration":
        container_name = "pulse_durations"
    else:
        raise ValueError(f"Unknown scan_type: {scan_type!r}")

    container = _get_attr_or_key(nv_sig, container_name, None)

    if container is None:
        container = {VirtualLaserKey.SCC: value}

        # Normal NVSig/dataclass path.
        try:
            setattr(nv_sig, container_name, container)
            return True
        except Exception:
            pass

        # Mapping-like fallback.
        try:
            nv_sig[container_name] = container
            return True
        except Exception:
            return False

    # Existing container.
    try:
        container[VirtualLaserKey.SCC] = value
        return True
    except Exception:
        pass

    # Some serialized structures may use string enum keys.
    for key in [
        getattr(VirtualLaserKey.SCC, "name", None),
        getattr(VirtualLaserKey.SCC, "value", None),
        "SCC",
        "scc",
    ]:
        if key is None:
            continue
        try:
            container[key] = value
            return True
        except Exception:
            pass

    return False


def update_nv_list_with_scc_optima(
    nv_list,
    scan_type,
    optimum_by_nv,
):
    """
    Update all NVSig objects with their fitted SCC optimum.

    Returns
    -------
    num_updated : int
        Number of NV entries successfully updated.
    """
    num_updated = 0

    for nv_ind, nv_sig in enumerate(nv_list):
        if nv_ind >= len(optimum_by_nv):
            break

        value = optimum_by_nv[nv_ind]

        if set_nv_scc_parameter(
            nv_sig,
            scan_type,
            value,
        ):
            num_updated += 1

    return num_updated


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_scc_sweep(
    data,
    local_fit_points,
    max_vertex_uncertainty_fraction,
    update_nv_sig,
):
    config_info = get_scc_config_info()
    step_vals, step_key = get_scan_values(data)
    scan_type, detection_source = determine_varied_parameter(data, step_vals)

    nv_list = data["nv_list"]

    counts = np.asarray(data["counts"])
    sig_counts = np.asarray(counts[0], dtype=np.float32)
    ref_counts = np.asarray(counts[1], dtype=np.float32)

    avg_snr, avg_snr_ste = widefield.calc_snr(sig_counts, ref_counts)
    avg_snr = np.asarray(avg_snr, dtype=float)
    avg_snr_ste = np.asarray(avg_snr_ste, dtype=float)

    if avg_snr.ndim != 2:
        raise ValueError(f"Expected avg_snr shape (num_nvs, num_steps); got {avg_snr.shape}")

    if avg_snr.shape[0] != len(nv_list):
        raise ValueError(
            f"Experiment has {len(nv_list)} NVs but SNR has {avg_snr.shape[0]} rows."
        )

    if avg_snr.shape[1] != step_vals.size:
        raise ValueError(
            f"Experiment has {step_vals.size} sweep values but SNR has {avg_snr.shape[1]} columns."
        )

    estimates = []

    for nv_ind in range(len(nv_list)):
        est = estimate_peak(
            step_vals,
            avg_snr[nv_ind],
            avg_snr_ste[nv_ind],
            local_fit_points=local_fit_points,
            max_vertex_uncertainty_fraction=max_vertex_uncertainty_fraction,
        )

        if scan_type == "duration" and np.isfinite(est.x_opt):
            est.x_opt = float(round(est.x_opt / 4.0) * 4.0)

        estimates.append(est)

    optimum_by_nv = np.array([est.x_opt for est in estimates], dtype=float)

    median_snr = np.nanmedian(avg_snr, axis=0)

    n_eff = np.sum(np.isfinite(avg_snr), axis=0)
    median_ste = np.nanmedian(avg_snr_ste, axis=0) / np.sqrt(np.maximum(n_eff, 1))
    median_ste = safe_sigma(median_ste)

    ensemble_estimate = estimate_peak(
        step_vals,
        median_snr,
        median_ste,
        local_fit_points=local_fit_points,
        max_vertex_uncertainty_fraction=max_vertex_uncertainty_fraction,
    )

    continuous_ensemble_optimum = float(ensemble_estimate.x_opt)

    if scan_type == "duration" and np.isfinite(ensemble_estimate.x_opt):
        ensemble_estimate.x_opt = float(round(ensemble_estimate.x_opt / 4.0) * 4.0)

    median_individual = float(np.nanmedian(optimum_by_nv))
    if scan_type == "duration":
        median_individual = float(round(median_individual / 4.0) * 4.0)

    base_sample = config_info["aod_base_sample"]

    if scan_type == "amplitude":
        physical_step_vals = base_sample * step_vals
        ensemble_physical = base_sample * ensemble_estimate.x_opt
        median_individual_physical = base_sample * median_individual
        physical_by_nv = base_sample * optimum_by_nv
        varied_parameter_label = "SCC AOD amplitude multiplier"
        physical_parameter_label = "Actual SCC AOD waveform sample"
    else:
        physical_step_vals = step_vals.copy()
        ensemble_physical = ensemble_estimate.x_opt
        median_individual_physical = median_individual
        physical_by_nv = optimum_by_nv.copy()
        varied_parameter_label = "SCC pulse duration (ns)"
        physical_parameter_label = "SCC pulse duration (ns)"

    # Preserve the values that were already stored in the incoming nv_sig list.
    # This is useful for comparing a new amplitude calibration to the previous one.
    nv_saved_series_before = gather_nv_saved_series(nv_list)

    num_nv_sig_updated = 0
    if update_nv_sig:
        num_nv_sig_updated = update_nv_list_with_scc_optima(
            nv_list,
            scan_type,
            optimum_by_nv,
        )

    # Re-read after updating so the analysis output also records what is now
    # present in nv_sig.
    nv_saved_series_after = gather_nv_saved_series(nv_list)

    results = {
        "scan_type": scan_type,
        "detection_source": detection_source,
        "experiment_step_key": step_key,
        "experiment_step_values": step_vals.tolist(),
        "physical_step_values": physical_step_vals.tolist(),
        "varied_parameter_label": varied_parameter_label,
        "physical_parameter_label": physical_parameter_label,
        "config_scc_physical_laser": config_info["physical_laser_name"],
        "config_scc_default_duration_ns": config_info["default_duration_ns"],
        "config_scc_aod_element": config_info["aod_element"],
        "config_scc_aod_operation": config_info["aod_operation"],
        "config_scc_aod_pulse_name": config_info["aod_pulse_name"],
        "config_scc_aod_waveform_name": config_info["aod_waveform_name"],
        "config_scc_aod_base_sample": base_sample,
        "ensemble_optimum_continuous": continuous_ensemble_optimum,
        "ensemble_optimum": float(ensemble_estimate.x_opt),
        "ensemble_optimum_physical": float(ensemble_physical),
        "median_individual_optimum": float(median_individual),
        "median_individual_optimum_physical": float(median_individual_physical),
        "optimal_value_by_nv": {
            int(i): float(optimum_by_nv[i]) if np.isfinite(optimum_by_nv[i]) else None
            for i in range(len(nv_list))
        },
        "optimal_physical_value_by_nv": {
            int(i): float(physical_by_nv[i]) if np.isfinite(physical_by_nv[i]) else None
            for i in range(len(nv_list))
        },
        "fit_status_by_nv": {
            int(i): estimates[i].status for i in range(len(nv_list))
        },
        "fit_method_by_nv": {
            int(i): estimates[i].method for i in range(len(nv_list))
        },
        "optimum_uncertainty_by_nv": {
            int(i): float(estimates[i].x_unc) if np.isfinite(estimates[i].x_unc) else None
            for i in range(len(nv_list))
        },
        "update_nv_sig": bool(update_nv_sig),
        "num_nv_sig_updated": int(num_nv_sig_updated),

        # Keep the actual NVSig objects in the saved analysis result so that,
        # when SAVE_RESULTS=True, the optimized SCC values persist.
        "nv_list": nv_list,

        "nv_saved_series_before": {
            series.name: {
                "label": series.label,
                "source": series.source,
                "values": series.values.tolist(),
            }
            for series in nv_saved_series_before
        },

        "nv_saved_series_after": {
            series.name: {
                "label": series.label,
                "source": series.source,
                "values": series.values.tolist(),
            }
            for series in nv_saved_series_after
        },
    }

    aux = {
        "scan_type": scan_type,
        "detection_source": detection_source,
        "config_info": config_info,
        "step_vals": step_vals,
        "physical_step_vals": physical_step_vals,
        "avg_snr": avg_snr,
        "avg_snr_ste": avg_snr_ste,
        "median_snr": median_snr,
        "median_ste": median_ste,
        "estimates": estimates,
        "ensemble_estimate": ensemble_estimate,
        "continuous_ensemble_optimum": continuous_ensemble_optimum,
        "optimum_by_nv": optimum_by_nv,
        "physical_by_nv": physical_by_nv,
        "nv_saved_series_before": nv_saved_series_before,
        "nv_saved_series_after": nv_saved_series_after,
        "num_nv_sig_updated": num_nv_sig_updated,
    }

    return results, aux


# =============================================================================
# PRINTING
# =============================================================================

def print_summary(results):
    print()
    print("============================================================")
    print("AUTOMATIC SCC PARAMETER-SWEEP ANALYSIS")
    print("============================================================")
    print("Varied parameter :", results["scan_type"])
    print("Detected from    :", results["detection_source"])

    steps = np.asarray(results["experiment_step_values"], dtype=float)
    print(
        "Measured sweep   : "
        f"{np.nanmin(steps):.6g} -> {np.nanmax(steps):.6g} ({steps.size} points)"
    )

    print()
    print("--- Active config ---")
    print("SCC physical laser      :", results["config_scc_physical_laser"])
    print(
        "Default SCC duration    : "
        f"{results['config_scc_default_duration_ns']:.6g} ns"
    )
    print("SCC AOD element         :", results["config_scc_aod_element"])
    print("SCC AOD pulse           :", results["config_scc_aod_pulse_name"])
    print("SCC AOD waveform        :", results["config_scc_aod_waveform_name"])
    print(
        "SCC AOD base sample     : "
        f"{results['config_scc_aod_base_sample']:.6g}"
    )

    print()
    print("--- Recommended global setting ---")
    if results["scan_type"] == "amplitude":
        print(
            "Continuous optimum      : "
            f"{results['ensemble_optimum_continuous']:.6f}"
        )
        print(
            "Optimum multiplier      : "
            f"{results['ensemble_optimum']:.6f}"
        )
        print(
            "Actual OPX sample       : "
            f"{results['ensemble_optimum_physical']:.6f}"
        )
    else:
        print(
            "Continuous optimum      : "
            f"{results['ensemble_optimum_continuous']:.3f} ns"
        )
        print(
            "Hardware setting        : "
            f"{results['ensemble_optimum']:.3f} ns"
        )

    print()
    print("--- Median of individual NV optima ---")
    if results["scan_type"] == "amplitude":
        print(
            "Median multiplier       : "
            f"{results['median_individual_optimum']:.6f}"
        )
        print(
            "Median actual sample    : "
            f"{results['median_individual_optimum_physical']:.6f}"
        )
    else:
        print(
            "Median SCC duration     : "
            f"{results['median_individual_optimum']:.3f} ns"
        )

    statuses = list(results["fit_status_by_nv"].values())
    status_names, counts = np.unique(statuses, return_counts=True)

    print()
    print("--- Per-NV fit status ---")
    for name, count in zip(status_names, counts):
        print(f"{name:30s}: {count}")

    print()
    print("--- NVSig update ---")
    print("Update requested          :", results["update_nv_sig"])
    print(
        "NVs updated              : "
        f"{results['num_nv_sig_updated']} / "
        f"{len(results['optimal_value_by_nv'])}"
    )

    for which in ["before", "after"]:
        nv_saved_series = results.get(
            f"nv_saved_series_{which}",
            {},
        )

        print()
        print(
            f"--- Per-NV saved amplitude/weight series "
            f"{which.upper()} update ---"
        )

        if len(nv_saved_series) == 0:
            print("None found with the current candidate field names.")
            continue

        for name, info in nv_saved_series.items():
            vals = np.asarray(
                info["values"],
                dtype=float,
            )
            finite = vals[np.isfinite(vals)]

            if finite.size:
                print(
                    f"{name:35s}: "
                    f"N={finite.size}, "
                    f"median={np.nanmedian(finite):.6g}, "
                    f"min={np.nanmin(finite):.6g}, "
                    f"max={np.nanmax(finite):.6g}, "
                    f"source={info['source']}"
                )
            else:
                print(
                    f"{name:35s}: no finite values"
                )

    print("============================================================")


# =============================================================================
# PLOTS
# =============================================================================

def plot_results(aux):
    scan_type = aux["scan_type"]
    config_info = aux["config_info"]
    step_vals = aux["step_vals"]
    avg_snr = aux["avg_snr"]
    median_snr = aux["median_snr"]
    median_ste = aux["median_ste"]
    estimate = aux["ensemble_estimate"]
    optimum_by_nv = aux["optimum_by_nv"]
    nv_saved_series_before = aux["nv_saved_series_before"]
    nv_saved_series_after = aux["nv_saved_series_after"]

    # -------------------------------------------------------------------------
    # Figure 1: median SNR curve
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        step_vals,
        median_snr,
        yerr=median_ste,
        fmt="o",
        capsize=3,
        label="Median SNR across NVs",
    )

    if estimate.fit_x is not None:
        ax.plot(estimate.fit_x, estimate.fit_y, label="Local weighted quadratic")

    ax.axvline(
        estimate.x_opt,
        linestyle="--",
        label=f"Global optimum = {estimate.x_opt:.4g}",
    )

    if scan_type == "amplitude":
        ax.set_xlabel("SCC AOD amplitude multiplier")

        base_sample = config_info["aod_base_sample"]

        def factor_to_sample(x):
            return np.asarray(x) * base_sample

        def sample_to_factor(x):
            return np.asarray(x) / base_sample

        secondary = ax.secondary_xaxis(
            "top", functions=(factor_to_sample, sample_to_factor)
        )
        secondary.set_xlabel("Actual SCC AOD waveform sample")
    else:
        ax.set_xlabel("SCC pulse duration (ns)")

    ax.set_ylabel("SCC SNR")
    ax.set_title(f"SCC {scan_type} optimization")
    ax.grid(alpha=0.25)
    ax.legend()

    # -------------------------------------------------------------------------
    # Figure 2: histogram of per-NV optimums
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    finite = optimum_by_nv[np.isfinite(optimum_by_nv)]

    if finite.size:
        bins = min(30, max(8, int(np.sqrt(finite.size))))
        ax.hist(finite, bins=bins)
        median_optimum = float(np.nanmedian(finite))
        ax.axvline(median_optimum, linestyle="--", label=f"Median = {median_optimum:.4g}")

    if scan_type == "amplitude":
        ax.set_xlabel("Optimal SCC AOD multiplier")
    else:
        ax.set_xlabel("Optimal SCC duration (ns)")

    ax.set_ylabel("NV count")
    ax.set_title("Distribution of per-NV optima")
    ax.grid(alpha=0.25)
    ax.legend()

    # -------------------------------------------------------------------------
    # Figure 3: optimum vs measured peak SNR
    # -------------------------------------------------------------------------
    best_measured_snr = np.nanmax(avg_snr, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(optimum_by_nv, best_measured_snr, alpha=0.65)

    if scan_type == "amplitude":
        ax.set_xlabel("Optimal SCC AOD multiplier")
    else:
        ax.set_xlabel("Optimal SCC duration (ns)")

    ax.set_ylabel("Best measured SCC SNR")
    ax.set_title("Per-NV optimum vs measured SCC SNR")
    ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # Figure 4: per-NV optimum by NV index
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    nv_index = np.arange(len(optimum_by_nv))
    ax.plot(nv_index, optimum_by_nv, marker="o", linestyle="-", linewidth=1)

    if scan_type == "amplitude":
        ax.set_ylabel("Optimal SCC AOD multiplier")
        ax.set_title("Per-NV fitted SCC amplitude optimum")
    else:
        ax.set_ylabel("Optimal SCC duration (ns)")
        ax.set_title("Per-NV fitted SCC duration optimum")

    ax.set_xlabel("NV index")
    ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # Figures 5+: values already stored in nv_sig before this analysis.
    # -------------------------------------------------------------------------
    for series in nv_saved_series_before:
        vals = np.asarray(
            series.values,
            dtype=float,
        )
        nv_index = np.arange(len(vals))
        finite = vals[np.isfinite(vals)]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            nv_index,
            vals,
            marker="o",
            linestyle="-",
            linewidth=1,
        )
        ax.set_xlabel("NV index")
        ax.set_ylabel(series.label)
        ax.set_title(
            f"{series.label} vs NV index (before update)"
        )
        ax.grid(alpha=0.25)

        if finite.size:
            median_val = float(
                np.nanmedian(finite)
            )
            ax.axhline(
                median_val,
                linestyle="--",
                label=f"Median = {median_val:.4g}",
            )
            ax.legend()

        fig, ax = plt.subplots(figsize=(7, 5))
        if finite.size:
            bins = min(
                30,
                max(
                    8,
                    int(np.sqrt(finite.size)),
                ),
            )
            ax.hist(
                finite,
                bins=bins,
            )
            ax.axvline(
                float(np.nanmedian(finite)),
                linestyle="--",
                label=(
                    f"Median = "
                    f"{np.nanmedian(finite):.4g}"
                ),
            )
            ax.legend()

        ax.set_xlabel(series.label)
        ax.set_ylabel("NV count")
        ax.set_title(
            f"{series.label} distribution "
            f"(before update)"
        )
        ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # Updated SCC values now stored in nv_sig.
    # Only plot SCC-related series here to avoid duplicating unrelated weights.
    # -------------------------------------------------------------------------
    for series in nv_saved_series_after:
        label_lower = series.label.lower()

        if (
            "scc" not in label_lower
            or "amplitude" not in label_lower
        ):
            continue

        vals = np.asarray(
            series.values,
            dtype=float,
        )
        nv_index = np.arange(len(vals))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            nv_index,
            vals,
            marker="o",
            linestyle="-",
            linewidth=1,
        )
        ax.set_xlabel("NV index")
        ax.set_ylabel(series.label)
        ax.set_title(
            "SCC amplitudes stored in nv_sig "
            "after analysis"
        )
        ax.grid(alpha=0.25)

    # -------------------------------------------------------------------------
    # For amplitude scans, compare the PREVIOUS saved SCC amplitude against
    # the newly fitted per-NV optimum.
    # -------------------------------------------------------------------------
    if scan_type == "amplitude":
        saved_scc_before = None

        for series in nv_saved_series_before:
            label_lower = series.label.lower()

            if (
                "scc" in label_lower
                and "amplitude" in label_lower
            ):
                saved_scc_before = series
                break

        if saved_scc_before is not None:
            x = np.asarray(
                saved_scc_before.values,
                dtype=float,
            )
            y = np.asarray(
                optimum_by_nv,
                dtype=float,
            )

            mask = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            if np.sum(mask) >= 3:
                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )
                ax.scatter(
                    x[mask],
                    y[mask],
                    alpha=0.65,
                )
                ax.set_xlabel(
                    "Previously saved SCC amplitude"
                )
                ax.set_ylabel(
                    "New fitted SCC amplitude optimum"
                )
                ax.set_title(
                    "Previous vs new per-NV SCC amplitude"
                )
                ax.grid(alpha=0.25)


def plot_individual_examples(aux, max_plots):
    scan_type = aux["scan_type"]
    x = aux["step_vals"]
    avg_snr = aux["avg_snr"]
    avg_snr_ste = aux["avg_snr_ste"]
    estimates = aux["estimates"]

    num_to_plot = min(len(estimates), int(max_plots))

    for nv_ind in range(num_to_plot):
        est = estimates[nv_ind]

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.errorbar(
            x,
            avg_snr[nv_ind],
            yerr=avg_snr_ste[nv_ind],
            fmt="o",
            capsize=2,
            label="Data",
        )

        if est.fit_x is not None:
            ax.plot(est.fit_x, est.fit_y, label="Local fit")

        if np.isfinite(est.x_opt):
            ax.axvline(est.x_opt, linestyle="--", label=f"optimum={est.x_opt:.4g}")

        if scan_type == "amplitude":
            ax.set_xlabel("SCC AOD amplitude multiplier")
        else:
            ax.set_xlabel("SCC pulse duration (ns)")

        ax.set_ylabel("SCC SNR")
        ax.set_title(f"NV {nv_ind}: {est.status}")
        ax.grid(alpha=0.25)
        ax.legend()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    kpl.init_kplotlib()

    # =========================================================================
    # USER PARAMETERS
    # =========================================================================

    # Dataset to analyze.
    # file_stem = "2026_09_15-04_51_20-qnami-nv0_2026_02_20" #amp
    # file_stem = "2026_09_15-18_24_43-qnami-nv0_2026_02_20" #amp
    file_stem = "2026_09_16-03_10_44-qnami-nv0_2026_02_20"
    
    

    # Local peak fit.
    local_fit_points = 18
    max_vertex_uncertainty_fraction = 0.30
    update_nv_sig = False

    # Plotting.
    show_plots = True
    plot_individual_fits = True
    max_individual_plots = 12
    save_results = True
    save_basename = "scc_parameter_sweep_analysis_with_nv_amps"

    # =========================================================================
    # LOAD + ANALYZE
    # =========================================================================

    print(f"Loading data: {file_stem}")

    data = dm.get_raw_data(
        file_stem=file_stem,
        load_npz=True,
    )

    results, aux = analyze_scc_sweep(
        data,
        local_fit_points=local_fit_points,
        max_vertex_uncertainty_fraction=max_vertex_uncertainty_fraction,
        update_nv_sig=update_nv_sig,
    )

    print_summary(results)

    # =========================================================================
    # PLOTS
    # =========================================================================

    if show_plots:
        plot_results(aux)

    if plot_individual_fits:
        plot_individual_examples(
            aux,
            max_plots=max_individual_plots,
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    if save_results:
        timestamp = dm.get_time_stamp()

        file_path = dm.get_file_path(
            __file__,
            timestamp,
            save_basename,
        )

        dm.save_raw_data(
            results,
            file_path,
        )

        print()
        print(f"Saved analysis: {file_path}")

    kpl.show(block=True)
