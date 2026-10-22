# Cornell Experiment 0: phase-tracked virtual-Z mode (Purcell)

## Run a short test batch (no blocking plot)
In the normal Purcell control-panel session, after constructing `nv_list`:

```python
from majorroutines.widefield import arbitrary_gate_tomography

arbitrary_gate_tomography.main(
    nv_list=nv_list,
    num_reps=200,
    num_runs=10,
    uwave_ind_list=(1,),
    mode="virtual_z",
    show_plots=False,
)
```

For a direct matched comparison, repeat with `mode="projection"` using
the same NV list, microwave chain, and calibration. Do not substitute the
Cornell `sz` column for the projected-Z targets used by these modes.

To run in two batches, wrap the call in `for _ in range(2):` and optionally
take a widefield image immediately before each call. Main returns without
waiting for a Matplotlib window; raw counts and two figures are saved per call.

## Pulse order and interpretation

For each Cornell gate, three signal experiments plus the dual-rail SCC
reference are acquired. In signal runs:

1. Initialize NV in +Z.
2. Prepare +Z (no pulse), +X (phase-shifted pi/2), or +Y (phase-shifted pi/2).
3. Apply at most ONE quantized, resonant transverse microwave gate pulse.
4. Apply the SAME final +X pi/2 (IQ phase 0) for all gates and input states.
5. SCC/read out final Z population.

The fourth SCC reference is unchanged: alternating even-repetition ms=0 and
odd-repetition ms=1, without prep/gate/final projection.

The 10 gate outputs are compared against exact target projected-Z expectations
for each initial +Z, +X, +Y state, rather than the original Bloch XYZ from
an initial +Z state. `mode="projection"` uses three XYX/YXY physical gate
pulses; `mode="virtual_z"` uses one plus preparation-phase bookkeeping.
Both modes use the SAME final +X pi/2 readout.

The logical gate identity is U=Rz(a) Ry(b) Rz(c)
= R_(a+90deg)(b) Rz(a+c). The logical Rz(a+c) is transferred into the phase
of the **known-state preparation pulses**: prepare +X at phase 90+a+c, +Y
at 180+a+c. This is sufficient for this *specific prepared-input experiment*.
It is NOT a physically independent arbitrary SU(2) gate on an unknown input
unless the logical Z frame is tracked in all surrounding operations.

## Current compilation restrictions

- 9-degree IQ phase grid from config/purcell.py (the hardware waveforms are
  not continuously programmable by this sequence).
- 4 ns OPX pulse-duration steps, minimum physical pulse length 16 ns.
- Pi and pi/2 durations from virtual signal generator config (no hardcoding).
- A short Euler beta rotation may be compiled as a long complementary rotation
  to meet the 16 ns minimum; fewer pulses therefore need not mean less MW time.
- Preflight verifies phase operations and rejects >0.15 rad computed
  full SO(3) gate error unless the user explicitly overrides the threshold.
- The saved `purcell_uwave1_virtual_z.json` manifest records all preparation
  phases, pulse phase and duration, logical-frame shift, theoretical responses,
  quantized-response predictions, errors, and the 3-pulse baseline.

Calibration caution: SG394 IQ phases, transients, and relative spin frame
must be checked experimentally. The offline predictions do NOT establish
hardware fidelity. The repo's own generated I/Q waveforms have nominal
phase samples; they are not a QUA oscillator `frame_rotation_2pi()`.

## Regression test (does not acquire data)
```powershell
cd C:\Users\matth\GitHub\dioptric
python -m unittest tests.test_purcell_virtual_z -v
```

Reanalyze a saved file with `analysis/arbitrary_gate_tomography.py`
using its file stem and reference quality cuts. Projection responses are
reported in the order initial +Z, +X, +Y.
