# Purcell fixed-T 13C nuclear-bath randomized-multisine sensing

## Separation from classical RQS
The existing `majorroutines/widefield/rqs_multisine.py` and older
`rqs_c13.py`, `rqs_c13_random_x.py` are unchanged.

**New acquisition:**
`majorroutines/widefield/rqs_c13_bath.py`
**New QUA:**
`servers/timing/sequencelibrary/QM_opx/camera/rqs_c13_bath.py`
**NVOffice analysis:**
`analysis/sc_rqs_c13_bath_analysis.py`
**NVOffice quantum/filter models:**
`analysis/rqs_c13_bath_models.py`

This experiment targets a stochastic/unpolarized 13C bath, not the classical
phase-locked coherent Fourier-amplitude estimation theorem. The multisine
compiler is retained only to generate diverse sign-changing pulse filters.
Every random setting has one fixed T and a variable number of calibrated
pi_X microwave pulses; no X/Y quadrature mixing or source phase locking.
Each camera acquisition uses SCC with alternating bright/dark reference.

## Fixed-T protocols
At expected B=31.1 G, bare 13C Larmor frequency is 33.29255 kHz.
Suggested exploratory grids (tone centers set by T):
- T=40 us: 8.0, 33.29255, 58.5851 kHz
- T=60 us: 15.0, 33.29255, 51.6 kHz
- T=80 us: 20.0, 33.29255, 46.6 kHz

The classical fixed-T Fourier resolution is ~1/T; the 3-bin spectral-power
screen is necessarily COARSE. Hyperfine features are not identified by bin
maxima. At T=60 us, periodic XY4 with 4 pi targets about 33.3 kHz and
periodic XY8 with 8 pi targets 66.7 kHz; both are saved at same T.
Also includes Hahn echo at same T.

## Safe no-hardware validation on Purcell
From dioptric repo root:
```powershell
python tests\test_rqs_c13_bath.py
python tests\test_rqs_c13_bath_full.py
python -m majorroutines.widefield.rqs_c13_bath --dry-run --T-us 60 --n-masks 16 --bank-size 128
```
Tests compile real QUA program objects; they do NOT execute microwave or
SCC pulses or open a hardware acquisition.

## Calibrated acquisition from existing Purcell control panel
```python
from majorroutines.widefield import rqs_c13_bath
raw = rqs_c13_bath.main(
    nv_list=nv_list, field_g=31.1,
    duration_us=60.0, seed=37,
    n_masks=64, bank_size=8192,
    uwave_ind_list=(1,),
    num_reps=50, num_runs=40,
)
```
This generates 64 randomized filters + XY8 + XY4 + echo = 67 logical
settings. Each setting gets 2000 signal SCC frames plus 2000 dual-rail
reference frames, total ~268000 camera exposures plus overhead. All
119 NVs are imaged simultaneously.

For a short device/SCC pilot: n_masks=16, num_reps=20, num_runs=10.
For a T sweep, run SEPARATE files at duration_us=40., 60., and 80.; the
NVOffice script REFUSES to merge different T or different fields.
Only use freshly calibrated pi/pi2 widths, field direction/strength, and
stable reference SCC. field_g is metadata, NOT a magnet-control command.

## NVOffice analysis
```powershell
python analysis\sc_rqs_c13_bath_analysis.py
python analysis\sc_rqs_c13_bath_analysis.py --files STEM_A STEM_B
python analysis\sc_rqs_c13_bath_analysis.py --per-nv 0
python analysis\sc_rqs_c13_bath_analysis.py --quantum-nv 15
```
No-argument form automatically selects latest completed data on NAS.
Unmatched filters from new seeds are retained; identical masks are
inverse-variance averaged AFTER independent SCC normalization.

Outputs via dm to configured NAS, with G: -> UNC fallback:
1) Filter bank |Y(f)|^2 vs frequency
2) Measured C across masks/NVs and varying pulse counts
3) Residual coherence versus 33kHz filter overlap
4) PSD-proxy coefficients, residual RMS
5) Held-out-mask prediction vs pulse-count-only null
6) Fixed-T XY4/XY8/Hahn coherence + SCC QC
7) Detailed per-NV multi-page PDF (12 NV default, all with --per-nv 0)
8) CSV/JSON and NPZ numeric outputs.
Optional selected-NV conditional single-13C simulator/fit: exploratory
only; requires independent physical controls and many-spin validation.

## Scientific and hardware limitations
- Original RQS *linear Fourier-amplitude estimator is invalid* for an
  unpolarized bath; this bath script does not use it.
- Gaussian PSD approximation can fail for individual strongly-coupled 13C.
- Conditional single-spin H0/h=fL Iz, H1/h=(fL+Apar)Iz+Aperp Ix model
  assumes instantaneous electron pi pulses and an unpolarized 13C.
- The finite 96 ns pulse phases/amplitudes, IQ projection sign, initial
  state and readout convention require hardware calibration.
- With x8 π at fixed T, a spectral coefficient can be confounded by
  pulse-count-dependent errors; compare held-out mask improvements and
  randomized XY8 controls. Same-T XY4 is the near-33kHz control at 60us.
- No hardware acquisition was launched during development.
