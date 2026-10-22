# Purcell: ^13C randomized-filter spectroscopy pilot (2026-10-22)

**Status:** Implemented as separate files. Offline schedule tests and QUA-program
construction passed. **NO physical acquisition was launched.** No existing
spin_echo, XY8, SCC, or config files were changed.

## Goal / physics

Test whether randomized timing of the electron-spin pi pulses reveals the
^13C-dependent NV decoherence/filter overlap while distributing sensitivity
across an otherwise unknown nuclear/hyperfine spectrum.

For ordinary, weak, stationary Gaussian magnetic noise:
```
C_r ~= C_base(T_r) exp(-chi_r)
chi_r proportional to integral [S_B(f) |Y_r(f)|^2] df
Y_r(f) = integral_0^T y_r(t) exp(i 2 pi f t) dt
```
Random filters yield different linear functionals of a *power spectrum* through
chi. The coherence is not linear in the unknown AC amplitude. For strongly
coupled ^13C, use the conditional nuclear-spin Hamiltonian in
`analysis/rqs_c13_quantum_sim.py` instead of interpreting chi as a classical
field PSD. This protocol is **RQS-inspired randomized DD**, not a faithful
implementation or speedup demonstration of Kannan's full optimal RQS or
noncommuting pooled-spectroscopy protocol.

The expected bare ^13C frequency is
`f_L = (1.0705 kHz/G)*B_parallel_or_magnitude`.
At **illustrative** B=33.4 G, f_L~35.76 kHz and an XY8-8 probe is centered
near it for full sensing time T ~ 8/(2*f_L) =112 us. This B value is inferred
from previous echo context, NOT read from Purcell's live magnet. Verify actual
field and the relevant NV orientation before acquiring. Nearby ^13C spins
can have large electron-state-conditional hyperfine shifts; a peak at
35.76 kHz is not guaranteed.

## Files

- `majorroutines/widefield/rqs_c13.py` host builder + acquisition + safe CLI
- `servers/timing/sequencelibrary/QM_opx/camera/rqs_c13.py` QUA
- `analysis/rqs_c13_analysis.py` SCC-normalized contrast + filter plot
- `analysis/rqs_c13_quantum_sim.py` joint 2x2 nuclear spin simulation
- `tests/test_rqs_c13_purcell.py` offline schedule, frequency, QUA build tests

## Sequence

For every step:
```
green prep + spin pol (existing base_scc)
pi/2_X - w0 - pi_X - w1 - pi_Y - w2 - pi_X - w3 - pi_Y
       - w4 - pi_Y - w5 - pi_X - w6 - pi_Y - w7 - pi_X
       - w8 - pi/2_X
SCC -> NUVU camera -> pause/trigger (existing base_scc)
dual-rail reference experiment (even rep = bright, odd = pi-dark)
```
Same 8 pi phases, 8 pi pulses, SCC reference, and π calibration in every
step. Random arm: pseudo-random positive wait partitions with a fixed T.
XY8 comparison arm: equal center-to-center pi separation. Both use **actual**
4ns clock-cycle quantized wait durations, pi 88ns / pi2 44ns from virtual
siggen 0 as of inspection, and record realized pulse center times.

Low bank: 40 random steps across T=64,96,128,160 us + 14 XY8 steps,
25-120 kHz. High bank: 40 random across T=8,12,20,32 us + 16 XY8 steps,
150 kHz-1.5 MHz. High bank is for **hyperfine-shifted** nuclear features,
not bare ^13C at ~33G. This selection may need change after initial data.

## Offline checks (safe, never touch lab)

Run from `C:\Users\matth\GitHub\dioptric`:
```powershell
python -m majorroutines.widefield.rqs_c13 --dry-run --band low --field-g 33.4
python -m majorroutines.widefield.rqs_c13 --dry-run --band high --field-g 33.4 --outdir analysis/rqs_c13_preview_high
python tests/test_rqs_c13_purcell.py
python analysis/rqs_c13_analysis.py --self-test
python analysis/rqs_c13_quantum_sim.py
```
The dry runs export exact metadata, each pi-pulse center/phase, and each of
the nine QUA wait durations. Compare representative pulses to timing traces
on the *actual* OPX before starting a many-NV experiment; QUA-program
construction is NOT a hardware timing characterization.

## Physical acquisition (manual only)

Start with a small, *current* calibrated NV list for just one microwave
orientation; check that microwave phase-90 pulse works and spin resonance
is still on frequency. Select a ^13C candidate field from a recent
spin-echo revival or independent magnetometer. In your existing live lab Python
session:
```python
from majorroutines.widefield import rqs_c13
raw = rqs_c13.main(
    nv_list=nv_list,      # your CURRENT calibrated subset, e.g. 10-20 NVs
    field_g=33.4,        # REPLACE with verified field for that NV orientation
    band="low",
    uwave_ind_list=(0,),
    masks_per_duration=10,
    num_reps=6,
    num_runs=3,
)
```
The low default is 54 steps x 6 repetitions x 3 runs x 2 experiments =
1,944 acquired SCC images (not including optional charge verification).
Run a tiny trial first using masks_per_duration=2, num_reps=2, num_runs=1,
inspect the raw counts and pulse timing, then scale up. The base routine
handles drift compensation, SCC, camera and current calibration.

The host always saves **raw data before analysis**, including all random
masks, explicit pulse timing, reference parity, random seed, hash, counts,
and complete/incomplete run status. An incomplete run is flagged and NOT
analyzed as complete. The underlying base_routine may return partial arrays
after an exception; do not treat uncompleted steps as valid measurements.

## Analyze after acquisition (saves directly to G: via data_manager)

Run from your normal Purcell terminal with the NAS mapped as G:.

```powershell
python analysis/rqs_c13_analysis.py --latest
# Or specify the exact acquisition file stem, WITHOUT the .txt/.npz extension:
python analysis/rqs_c13_analysis.py --file-stem 2026_10_22-12_00_00-johnson-nv0_2026_10_02
# High band requires a chosen hyperfine candidate:
python analysis/rqs_c13_analysis.py --file-stem YOUR_FULL_STEM --target-khz 340
```

By default, the four outputs are saved using `dm.get_file_path`,
`dm.save_figure`, `dm.save_text`, and `dm.save_npz` to:

`G:\nvdata\pc_Purcell\branch_master\rqs_c13_analysis\YYYY_MM\<acquisition_stem>\`

- `<analysis_timestamp>-rqs_c13_analysis.png` (diagnostic plot)
- `<analysis_timestamp>-rqs_c13_analysis.csv` (per-NV fit table)
- `<analysis_timestamp>-rqs_c13_analysis.json` (source/method/QC metadata)
- `<analysis_timestamp>-rqs_c13_analysis.npz` (coherence, reference, filters)

There is no automatic local fallback if the G: NAS is missing. To opt into
local output for debugging ONLY, provide `--outdir analysis/rqs_c13_results`.
The `--latest` option searches the latest month folder on the NAS.
The per-NV regression removes a separate duration-specific intercept
before testing if **dephasing** follows calculated near-13C filter overlap.
It is a qualitative diagnostic; it does not prove chemical identity,
infer calibrated PSD, or establish a quantum advantage. Confirm using
field-dependent scaling near 1.0705 kHz/G, persistence/reproducibility,
XY8 frequency sweeps, an off-band control, and (ideally) correlation
spectroscopy. Beware spurious multipulse harmonics.

## Next steps

1. Check measured B and field alignment; verify f_L vs measured spin echo.
2. OPX hardware timing trace/simulated samples for representative masks,
   including IQ ring-down, guard times, final projection sign and pulse phases.
3. Acquire low-bank mini-run on e.g. 10 good NVs, inspect bright/dark SCC
   separation, signs, and signal coherence without any inversion.
4. For MHz hyperfine candidates, run high band and simulate A_parallel/A_perp
   against field-dependent resonances; compare to XY8 controls.
5. Benchmark total wall time and confidence vs existing optimized
   spin-echo/XY8 and correlation sensing. No presumed N-fold speedup.
