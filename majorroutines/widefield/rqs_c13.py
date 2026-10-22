# -*- coding: utf-8 -*-
"""13C RQS-inspired randomized-filter spectroscopy on Purcell.

This acquisition compares RANDOM-timing XY8-8 and PERIODIC XY8-8 with the
same number of microwave pi pulses, SCC imaging, and interleaved dual-rail
reference. This is NOT Kannan's full optimal RQS/pooling theorem protocol.

No hardware is accessed by the CLI --dry-run path. For acquisition call main
with your already calibrated nv_list in an active Purcell lab session.

Important: strongly coupled 13C spins require a joint electron/nuclear model;
a classical PSD inversion is NOT automatically a hyperfine spectrum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from majorroutines.widefield import base_routine
from utils import data_manager as dm
from utils import tool_belt as tb
from utils import widefield

NUM_PI = 8
CLOCK_NS = 4
PHASES = (0, 90, 0, 90, 90, 0, 90, 0)
GAMMA_C13_HZ_PER_G = 1070.5  # approx: 10.705 MHz/T


def _integer_partition(total, bins, rng, min_cc=8):
    """Positive wait vector (clock cycles) with an exact clock-cycle sum."""
    spare = int(total) - bins * min_cc
    if spare < 0:
        raise ValueError("Duration too short for eight pi pulses.")
    weights = rng.dirichlet(np.full(bins, 1.2))
    extra = np.floor(spare * weights).astype(int)
    remainder = spare - int(extra.sum())
    if remainder:
        for j in rng.permutation(bins)[:remainder]:
            extra[j] += 1
    return (extra + min_cc).astype(int).tolist()


def _periodic_waits(total_cc, pi_cc, pi2_cc, min_cc=8):
    """Approximate equal PI-center spacings, corrected to exact total T."""
    total_wait = int(total_cc - NUM_PI * pi_cc - pi2_cc)
    middle = int(round(total_cc / NUM_PI - pi_cc))
    edge = int(round(total_cc / (2 * NUM_PI) - (pi_cc + pi2_cc) / 2))
    waits = [edge] + [middle] * (NUM_PI - 1)
    waits.append(total_wait - sum(waits))
    if min(waits) < min_cc:
        raise ValueError("XY8 center spacing is incompatible with pulse widths.")
    return waits


def _realized_pulse_centers_ns(waits_cc, pi_ns, pi2_ns):
    """Return pulse-center offsets relative to center of first pi/2."""
    t = pi2_ns  # elapsed from START of first pi/2
    centers = []
    for wait_cc in waits_cc[:-1]:
        t += CLOCK_NS * int(wait_cc)
        centers.append(t + pi_ns / 2 - pi2_ns / 2)
        t += pi_ns
    t += CLOCK_NS * int(waits_cc[-1])
    total_center_duration_ns = t + pi2_ns / 2 - pi2_ns / 2
    # t is time from start of first pi/2 to start of last pi/2.
    # Therefore t is also separation between the two pi/2 centers.
    return [float(x) for x in centers], float(t)


def _make_step(kind, t_ns, waits_cc, pi_ns, pi2_ns, target_f_hz=None):
    centers, realized_t_ns = _realized_pulse_centers_ns(
        waits_cc, pi_ns, pi2_ns
    )
    if kind not in {"random", "xy8"}:
        raise ValueError("Unsupported step kind")
    return {
        "kind": kind,
        "requested_T_ns": int(t_ns),
        "T_ns": realized_t_ns,
        "pi_centers_ns": centers,
        "waits_cc": [int(v) for v in waits_cc],
        "phase_deg": list(PHASES),
        "target_f_hz": (None if target_f_hz is None else float(target_f_hz)),
    }


def build_protocol(
    field_g=33.4,
    band="low",
    seed=13,
    masks_per_duration=10,
    pi_ns=88,
    pi2_ns=44,
):
    """Generate deterministic RQS/XY8 comparison schedules.

    field_g sets ONLY the EXPECTED 13C Larmor frequency. It does not set
    the lab field. For 'high', explore hyperfine-shifted components, not the
    bare 13C Larmor frequency. All waits are >= 32ns.
    """
    if not np.isfinite(field_g) or field_g <= 0:
        raise ValueError("field_g must be measured, finite and positive")
    if band not in ("low", "high"):
        raise ValueError("band must be low or high")
    if masks_per_duration < 2 or masks_per_duration > 64:
        raise ValueError("masks_per_duration outside [2,64]")
    if pi_ns % 4 or pi2_ns % 4 or pi_ns < 16 or pi2_ns < 16:
        raise ValueError("pi and pi/2 durations must be multiples of 4 ns")

    larmor_hz = field_g * GAMMA_C13_HZ_PER_G
    rng = np.random.default_rng(seed)
    pi_cc, pi2_cc = pi_ns // CLOCK_NS, pi2_ns // CLOCK_NS
    if band == "low":
        # At B~33 G, bare 13C is ~35.7 kHz. 8 pi pulses at this resonance
        # correspond to T~112 us. Frequencies outside this band may alias.
        random_T_us = (64, 96, 128, 160)
        xy_freqs = np.geomspace(25000, 120000, 14)
    else:
        # Shifted/strongly coupled 13C: an exploratory HIGH-frequency bank.
        random_T_us = (8, 12, 20, 32)
        xy_freqs = np.geomspace(150000, 1500000, 16)

    steps = []
    for t_us in random_T_us:
        t_ns = int(t_us * 1000)
        t_cc = t_ns // CLOCK_NS
        total_wait = t_cc - NUM_PI * pi_cc - pi2_cc
        for _ in range(masks_per_duration):
            waits = _integer_partition(total_wait, 9, rng)
            steps.append(_make_step("random", t_ns, waits, pi_ns, pi2_ns))

    for f_hz in xy_freqs:
        t_ns = int(round(NUM_PI * 1e9 / (2 * f_hz) / CLOCK_NS) * CLOCK_NS)
        t_cc = t_ns // CLOCK_NS
        waits = _periodic_waits(t_cc, pi_cc, pi2_cc)
        steps.append(_make_step("xy8", t_ns, waits, pi_ns, pi2_ns, f_hz))

    proto = {
        "experiment": "rqs_c13_random_timing_xy8_vs_periodic_xy8",
        "method": "RQS-inspired dephasing-filter comparison, not pooled quantum algorithm",
        "seed": int(seed),
        "band": band,
        "field_g_expected": float(field_g),
        "bare_c13_larmor_hz_expected": float(larmor_hz),
        "pi_ns": int(pi_ns),
        "pi2_ns": int(pi2_ns),
        "microwave_phases_deg": list(PHASES),
        "clock_ns": CLOCK_NS,
        "number_of_pi_pulses": NUM_PI,
        "masks_per_duration": int(masks_per_duration),
        "steps": steps,
    }
    serial = json.dumps(proto, sort_keys=True, separators=(",", ":"))
    proto["protocol_sha256"] = hashlib.sha256(serial.encode()).hexdigest()
    return proto


def export_protocol(proto, outdir):
    """Preview timing WITHOUT connecting to OPX, camera, or microwave servers."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rqs_c13_protocol.json").open("w", encoding="utf-8") as f:
        json.dump(proto, f, indent=2)
    with (out / "rqs_c13_pulses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "kind", "target_f_hz", "realized_T_ns",
                         "pulse", "center_ns", "phase_deg"])
        for ind, step in enumerate(proto["steps"]):
            for j, center in enumerate(step["pi_centers_ns"]):
                writer.writerow([ind, step["kind"], step["target_f_hz"],
                                 step["T_ns"], j + 1, center, PHASES[j]])
    with (out / "rqs_c13_waits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "kind", "T_ns"] +
                        [f"wait{j}_cc" for j in range(NUM_PI + 1)])
        for ind, step in enumerate(proto["steps"]):
            writer.writerow([ind, step["kind"], step["T_ns"], *step["waits_cc"]])
    return out


def main(
    nv_list,
    *,
    field_g=33.4,
    band="low",
    uwave_ind_list=(0,),
    seed=13,
    masks_per_duration=10,
    num_reps=6,
    num_runs=3,
):
    """Acquire the 13C spectroscopy experiment.

    Intentionally no automatically executed experiment. This call WILL operate
    hardware; run only during a controlled lab session with selected NVs.
    """


    if not nv_list:
        raise ValueError("Provide at least one calibrated NV signature")
    uwave_ind_list = list(uwave_ind_list)
    if len(uwave_ind_list) != 1 or uwave_ind_list[0] not in (0, 1, 2):
        raise ValueError("Start with exactly ONE microwave chain/orientation")
    if num_reps < 2 or num_reps % 2:
        raise ValueError("Use an even number of repetitions >= 2 for dual rail")
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    sig_cfg = tb.get_virtual_sig_gen_dict(uwave_ind_list[0])
    proto = build_protocol(
        field_g=field_g, band=band, seed=seed,
        masks_per_duration=masks_per_duration,
        pi_ns=int(sig_cfg["pi_pulse"]),
        pi2_ns=int(sig_cfg["pi_on_2_pulse"]),
    )
    print("13C expected bare Larmor: %.3f kHz; band: %s; steps: %d"
          % (proto["bare_c13_larmor_hz_expected"] / 1e3,
             band, len(proto["steps"])))
    base_args = widefield.get_base_scc_seq_args(nv_list, uwave_ind_list)
    pulse_gen = tb.get_server_pulse_gen()

    def run_fn(shuffled_step_inds):
        # ONE QUA program per run; no recompile per random mask.
        wait_tables = [
            [proto["steps"][i]["waits_cc"][j] for i in shuffled_step_inds]
            for j in range(NUM_PI + 1)
        ]
        seq_args = [base_args, wait_tables]
        pulse_gen.stream_load("rqs_c13.py", tb.encode_seq_args(seq_args), num_reps)

    raw_data = base_routine.main(
        nv_list=nv_list,
        num_steps=len(proto["steps"]),
        num_reps=num_reps,
        num_runs=num_runs,
        run_fn=run_fn,
        uwave_ind_list=uwave_ind_list,
        num_exps=2,
        ref_by_rep_parity=True,
        load_iq=True,
    )
    # SAVE RAW FIRST. Analysis is separate and cannot prevent raw-data saving.
    complete_runs = sum(x is not None for x in raw_data["step_ind_master_list"])
    raw_data.update({
        "timestamp": dm.get_time_stamp(),
        "rqs_c13_protocol": proto,
        "rqs_complete_runs": int(complete_runs),
        "rqs_expected_runs": int(num_runs),
        "rqs_complete": complete_runs == num_runs,
    })
    repr_nv_name = widefield.get_repr_nv_sig(nv_list).name
    path = dm.get_file_path(__file__, raw_data["timestamp"], repr_nv_name)
    dm.save_raw_data(raw_data, path)
    print("Raw data saved:", path)
    if not raw_data["rqs_complete"]:
        raise RuntimeError(
            "Partial acquisition: data was saved and explicitly flagged INCOMPLETE."
        )
    tb.reset_cfm()
    return raw_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Create and check schedules; never touch hardware")
    parser.add_argument("--field-g", type=float, default=33.4)
    parser.add_argument("--band", choices=("low", "high"), default="low")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--masks", type=int, default=10)
    parser.add_argument("--pi-ns", type=int, default=88)
    parser.add_argument("--pi2-ns", type=int, default=44)
    parser.add_argument("--outdir", default="analysis/rqs_c13_preview")
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("CLI only supports --dry-run; call main(nv_list=...) from lab")
    proto = build_protocol(args.field_g, args.band, args.seed, args.masks,
                           args.pi_ns, args.pi2_ns)
    outdir = export_protocol(proto, args.outdir)
    for step in proto["steps"]:
        assert len(step["waits_cc"]) == 9 and min(step["waits_cc"]) >= 8
        assert abs(step["T_ns"] - step["requested_T_ns"]) <= 4
    print("SAFE DRY RUN (no hardware).")
    print("Band:", args.band, "Bare 13C Larmor (kHz):",
          round(proto["bare_c13_larmor_hz_expected"] / 1e3, 3))
    print("Patterns:", len(proto["steps"]), "SHA:", proto["protocol_sha256"])
    print("Exported:", outdir.resolve())
