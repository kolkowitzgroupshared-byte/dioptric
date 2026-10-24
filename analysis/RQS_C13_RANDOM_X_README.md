# Random-pi_X spectroscopy pilot (October 2026)

The Purcell hardware code implements randomized timings of eight pi_X pulses,
with periodic XY8-8 controls and SCC dual-rail referencing. Existing
rqs_c13.py and spin_echo.py files are untouched.

Files: majorroutines/widefield/rqs_c13_random_x.py;
servers/timing/sequencelibrary/QM_opx/camera/rqs_c13_random_x.py;
tests/test_rqs_c13_random_x.py.

NVOffice estimator: analysis/sc_rqs_c13_random_x_estimator.py.

## Suggested first calibrated acquisition

```python
from majorroutines.widefield import rqs_c13_random_x
raw = rqs_c13_random_x.main(
    nv_list=nv_list, field_g=31.1, band="high",
    uwave_ind_list=(1,), seed=37, masks_per_duration=16,
    num_reps=50, num_runs=40, readout="cos",
)
```

80 settings total: 64 randomized + 16 XY8.
At 2,000 repetitions per setting with 2 SCC camera images per
repetition = 320,000 camera exposures. field_g is expected metadata,
not a magnet command.

Check fewer shots first: masks_per_duration=8, num_reps=50,
num_runs=10. This is suitable for hardware/QC checks, not a full 8-bin
NNLS reconstruction.

To compare with randomized XY8, run the ORIGINAL rqs_c13.main
with identical seed=37, masks_per_duration=16, NVs, field and shots.
To measure coherent AC quadratures, use readout="both" and calibrate
phase sign. For stochastic 13C, readout="cos" is appropriate.

Spin-echo shot-convergence data: median NV SEM ~0.100 at 1,800
signal repetitions per setting and ~0.051 at 7,200. This is only an
approximate guide for RQS, which has different timing/filter response.

The estimator outputs nonnegative spectral-weight proxies under
weak, stationary Gaussian-noise assumptions, NOT calibrated hyperfine
couplings. Strongly coupled nuclei require a joint electron-nuclear
model. This pilot does not claim Ishaan Kannan's optimal RQS theorem.
