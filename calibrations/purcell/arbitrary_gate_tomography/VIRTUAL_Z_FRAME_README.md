# Cornell Experiment 0 — one-pulse `virtual_z_frame`

**Installed on Purcell, October 23, 2026.** Dry-run software validated.
No live OPX acquisition was started during implementation.

## Intent

Compare Cornell's identical arbitrary SU(2) targets in:

- `mode="projection"`: conventional XYX/YXY three-pulse resonant gate,
  common initial preparations, physical phase-0 +X pi/2 projection.
- `mode="virtual_z_frame"`: *one* resonant gate IQ pulse; standard
  preparations and a frame-corrected final projection.
- `mode="virtual_z"`: older known-input **preparation-phase shortcut**
  remains intact as a historical control.

The four SCC channels are unchanged: +Z, +X, +Y signal preparations, then
the parity-alternating ms0/ms1 dual-rail reference. The reference has
neither a Cornell gate nor a projection pulse.

## Physical pulse sequence per signal channel

1. Polarize to +Z (optical).
2. Prepare **same as three-pulse baseline**:
   +Z: no microwave pulse
   +X: 48-ns pi/2 microwave pulse at 90 degrees
   +Y: 48-ns pi/2 microwave pulse at 180 degrees
3. Accumulate virtual Rz(gamma) in the software phase frame.
4. Play exactly ONE physical transverse microwave gate pulse at
   `phase = (90 - gamma) mod 360`, duration taken from compiler.
5. Accumulate virtual Rz(alpha); the final frame is
   `delta = (alpha + gamma) mod 360`.
6. Play a physical 48-ns pi/2 logical +X projection at the
   **frame-corrected** IQ phase `(-delta) mod 360`.
7. Spin-to-charge conversion, camera readout.

**Note:** "One physical pulse" refers only to the *arbitrary Cornell gate*.
Transverse state preparation and the analysis pi/2 remain physical pulses.

Mathematically, `U = Rz(alpha) Ry(beta) Rz(gamma)`. A persistent logical
frame is equivalent to the target SU(2) gate, but the current Experiment 0
implementation **consumes** the frame in the final projection. For longer
algorithms, all downstream gates and measurements must carry the frame.

This is a software phase accumulator adapted to the separate zero-IF
I- and Q-OPX elements on Purcell; no native `frame_rotation_2pi`
operation is applied to the individual analog channels.

The compiler uses the current config pi=96ns / pi/2=48ns, 4-ns timing,
16-ns min-pulse and the existing 1-degree IQ waveforms.
No power scaling, detuning, or OPX config change was made.

## Run offline manifest and validations

```python
from majorroutines.widefield import arbitrary_gate_tomography as cornell
manifest, path = cornell.prepare(uwave_ind=1, mode="virtual_z_frame")
for gate in manifest["gates"]:
    print(gate["gate"],
          [(p["phase_deg"], p["duration_ns"]) for p in gate["pulses"]],
          "projection phase:", gate["projection_phase_deg"])
```

```powershell
python -m unittest tests.test_purcell_virtual_z_frame tests.test_purcell_virtual_z -v
```

## Start a hardware acquisition — ONLY WHEN READY

Use the locked and already-validated NV list for your same comparison:

```python
from majorroutines.widefield import arbitrary_gate_tomography as cornell
raw = cornell.main(
    nv_list=nv_list,
    num_reps=200,
    num_runs=5,
    uwave_ind_list=(1,),
    mode="virtual_z_frame",
    show_plots=False,
)
```

Use identical `num_reps`, `num_runs`, `nv_list`, frequency,
power, and optical SCC configuration for the `mode="projection"` control.
Prefer interleaving short blocks (projection/frame/frame/projection)
rather than comparing only runs separated by hours.

## Multi-file comparison

Edit `FILE_STEMS` in `analysis/arbitrary_gate_multifile_analysis.py`:

```python
FILE_STEMS = [
    "2026_10_21-22_42_30-johnson-nv0_2026_10_02", # old 3-pulse projection
    "NEW_VIRTUAL_Z_FRAME_STEM",                     # new 1-pulse mode
]
```

Run `python -m analysis.arbitrary_gate_multifile_analysis`.
The analyzer detects `virtual_z_frame` separately, normalizes each
file independently, matches NVs by name, and compares equal Cornell
projected-Z targets. It prioritizes frame mode vs projection if all
three modes are listed.

## Diagnostics and cautions

- Gate-by-gate compilation and predicted measurements were verified in
  SciPy SO(3), and all ten QUA programs were *constructed*, not executed.
- The maximum predicted full-SO(3) compiler discrepancy at pi=96ns
  with 1-degree IQ phases is ~0.0618 rad; **not measured gate infidelity**.
- The logical projection is still +X pi/2, although its **physical**
  microwave phase changes with the gate.
- Control hardware phase offsets, I/Q imbalance, pulse transients,
  and the current SCC normalization require experimental calibration.
- Do not interpret fixed-one-axis, three-input projected-Z signals
  as complete quantum process tomography.
