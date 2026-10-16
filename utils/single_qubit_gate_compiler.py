"""Cornell arbitrary SU(2) gates: verified CSV targets and Purcell XYX/YXY compiler.

Rotation convention in Cornell_Berkeley_expV0.pdf:
U = exp[-i Omega * (sin(theta)sin(phi)*X + sin(theta)cos(phi)*Y
                     + cos(theta)*Z)].

The compiled sequence is a chronological list of resonant X/Y pulses. The
QM elements for the microwave chains in Purcell already support phase 0,
90, 180, 270 in config/purcell.py; these DO NOT require virtual-Z support.
Each pulse duration is quantized to the 4-ns OPX clock and audited against
the *full SO(3) gate*, not only the Bloch vector for |0>.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

# Experiment constants live here rather than in a separate config module.
# The physical IQ operations and MW pi durations still come from config/purcell.py.
# This is a compiler-only convention (no hardware reconfiguration).
CONFIG = {
    "phase_X_pos_deg": 0,
    "phase_X_neg_deg": 180,
    "phase_Y_pos_deg": 90,
    "phase_Y_neg_deg": 270,
    "basis_order": ["Z", "X", "Y"],
    "clock_ns": 4,
    "min_pulse_ns": 16,
    "max_rotation_error_rad": 0.15,
}


AXIS_VEC = {"X": np.array([1., 0., 0.]), "Y": np.array([0., 1., 0.])}


def source_rotation(row):
    th, ph, om = (float(row[k]) for k in ("theta", "phi", "Omega"))
    n = np.array([np.sin(th)*np.sin(ph), np.sin(th)*np.cos(ph), np.cos(th)])
    return Rotation.from_rotvec(2*om*n)


def load_cornell_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Cornell CSV contains no gates")
    required = {"Omega", "theta", "phi", "sx", "sy", "sz"}
    for i, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"Gate {i}: missing {required-set(row)}")
        pred = source_rotation(row).apply([0.,0.,1.])
        supplied = np.array([float(row[k]) for k in ("sx", "sy", "sz")])
        if np.max(np.abs(pred-supplied)) > 1e-6:
            raise ValueError(f"Gate {i+1} CSV convention mismatch: {pred} != {supplied}")
    return rows


def _pulse(axis, angle_rad, pi_ns, clock_ns, min_pulse_ns):
    """Compile a single resonant rotation R_axis(angle). 2pi fallback avoids <16ns pulses."""
    # Remove full SO(3) turns; global SU(2) signs are not observable here.
    angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
    if abs(angle_rad) < 1e-9:
        return None
    negative = angle_rad < 0
    physical_rad = abs(angle_rad)
    ideal_duration_ns = physical_rad * pi_ns / math.pi
    used_full_turn = False
    if ideal_duration_ns < min_pulse_ns:
        physical_rad = 2*math.pi-physical_rad
        negative = not negative
        ideal_duration_ns = physical_rad*pi_ns/math.pi
        used_full_turn = True
    cc = int(round(ideal_duration_ns/clock_ns))
    if cc * clock_ns < min_pulse_ns:
        raise ValueError("Pulse shorter than hardware minimum")
    if axis == "X":
        phase = CONFIG["phase_X_neg_deg" if negative else "phase_X_pos_deg"]
    else:
        phase = CONFIG["phase_Y_neg_deg" if negative else "phase_Y_pos_deg"]
    actual_angle = (-1 if negative else 1)*cc*clock_ns*math.pi/pi_ns
    return {"axis": axis, "phase_deg": phase, "duration_cc": cc,
            "duration_ns": cc*clock_ns, "requested_angle_rad": angle_rad,
            "actual_angle_rad": actual_angle, "full_turn_fallback": used_full_turn}


def _compile_euler(rot, order, pi_ns, clock_ns, min_pulse_ns):
    # SciPy capital-letter Euler convention is intrinsic. Rx(a)Ry(b)Rx(c)
    # is applied chronologically as c, b, a; test both XYX and YXY.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a, b, c = rot.as_euler(order, degrees=False)
    axes = [order[2], order[1], order[0]]
    pulses = [p for p in (_pulse(ax, angle, pi_ns, clock_ns, min_pulse_ns)
                          for ax, angle in zip(axes, (c,b,a))) if p is not None]
    realized = Rotation.identity()
    for p in pulses:
        realized = Rotation.from_rotvec(AXIS_VEC[p["axis"]]*p["actual_angle_rad"])*realized
    error = float((rot.inv()*realized).magnitude())
    return {"decomposition": order, "pulses": pulses,
            "rotation_error_rad": error,
            "duration_total_ns": sum(p["duration_ns"] for p in pulses),
            "predicted_actual_bloch": realized.apply([0,0,1]).tolist()}


def compile_gates(csv_path, pi_pulse_ns, mw_chain, max_error_rad=None):
    rows = load_cornell_csv(csv_path)
    pi_ns = float(pi_pulse_ns)
    clock = CONFIG["clock_ns"]
    minimum = CONFIG["min_pulse_ns"]
    if pi_ns < 2*minimum or pi_ns%clock:
        raise ValueError("MW pi_pulse_ns must be >=32 ns and a multiple of 4 ns")
    manifest = {
        "format": "cornell-arb-sqr-v1",
        "csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "mw_chain": int(mw_chain), "pi_pulse_ns": pi_ns,
        "pi_on_2_pulse_ns": pi_ns/2, "clock_ns": clock,
        "phase_available_deg": [0,90,180,270],
        "basis_order": CONFIG["basis_order"],
        "gates": []}
    for i,row in enumerate(rows,1):
        rot = source_rotation(row)
        candidates = [_compile_euler(rot, order, pi_ns, clock, minimum)
                      for order in ("XYX", "YXY")]
        # Prioritize precise full-gate implementation, use shorter as tie-breaker.
        best = min(candidates,key=lambda c:(round(c["rotation_error_rad"], 5), c["duration_total_ns"]))
        expected = [float(row[k]) for k in ("sx","sy","sz")]
        best.update({"gate":i,"Omega":float(row["Omega"]),
                     "theta":float(row["theta"]),"phi":float(row["phi"]),
                     "expected_bloch_xyz":expected})
        manifest["gates"].append(best)
    cutoff = CONFIG["max_rotation_error_rad"] if max_error_rad is None else max_error_rad
    manifest["max_rotation_error_rad"] = max(g["rotation_error_rad"] for g in manifest["gates"])
    manifest["hardware_warning"] = manifest["max_rotation_error_rad"] > cutoff
    return manifest


def save_manifest(manifest, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return path


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest, opx_config, chain, pi_ns, allow_large_error=False):
    """Fail before touching hardware if phase operations/calibration are stale."""
    if manifest["format"] != "cornell-arb-sqr-v1":
        raise ValueError("Unknown manifest format")
    if manifest["mw_chain"] != chain or abs(manifest["pi_pulse_ns"]-pi_ns)>1e-8:
        raise ValueError("Stale manifest: microwave chain or calibrated pi time changed; recompile")
    if manifest["hardware_warning"] and not allow_large_error:
        raise ValueError("Gate compilation error exceeds configured limit; inspect manifest before running")
    for phase in manifest["phase_available_deg"]:
        # Exactly one IQ operation per phase for each relevant microwave chain
        from utils import tool_belt as tb
        device = tb.get_virtual_sig_gen_dict(chain)["physical_name"]
        for comp in ("i","q"):
            element=f"ao_{device}_{comp}"
            operations=opx_config["elements"][element]["operations"]
            for pulse in ("pi_pulse","pi_on_2_pulse"):
                name=f"{pulse}_{phase}"
                if name not in operations:
                    raise ValueError(f"Missing OPX IQ operation {element}:{name}")


def main():
    p=argparse.ArgumentParser(description="Dry-run Cornell CSV -> Purcell gate manifest")
    p.add_argument("--csv",required=True)
    p.add_argument("--pi-ns",type=float,required=True)
    p.add_argument("--chain",type=int,default=0)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    m=compile_gates(a.csv,a.pi_ns,a.chain)
    save_manifest(m,a.out)
    print(f"Saved {len(m['gates'])} gates to {a.out}; worst SO(3) error {m['max_rotation_error_rad']:.5f} rad")
    for g in m["gates"]:
        print(f"gate {g['gate']:2d} {g['decomposition']}  "
              f"{[(p['phase_deg'],p['duration_ns']) for p in g['pulses']]} "
              f"error={g['rotation_error_rad']:.5f} rad")
    if m["hardware_warning"]:
        raise SystemExit("Compilation warning: gate error exceeds configured limit; DO NOT RUN")

if __name__=="__main__":
    main()
