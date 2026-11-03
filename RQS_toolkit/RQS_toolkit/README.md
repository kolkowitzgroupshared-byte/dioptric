# Randomized quantum spectroscopy (RQS) toolkit

Design random single-qubit pulse sequences for known signal frequencies, export
an acquisition plan for hardware, and decode the measured outcomes into the
signal's Fourier amplitudes. Linear observables are chosen **after** the data
are taken, and the same data can be reused for any number of them.

This folder is self-contained: the only package is `randomized_sensing`.

## Start

Requires Python 3.10+ with NumPy, SciPy, Matplotlib and Jupyter. Nothing needs
to be installed from this folder: the notebook loads `src/randomized_sensing`
directly. From this folder:

```sh
python -m pip install numpy scipy matplotlib jupyterlab
jupyter lab randomized_experiment.ipynb
```

To use the package from your own scripts, run them from this folder with
`PYTHONPATH=src`, or add `src` to `sys.path` as the notebook's first cell does.

Run the notebook from top to bottom; cells marked ✏️ hold the settings to edit.
It walks through describing the tones and hardware (step 1), building the random
pulse sequences (2), planning the shots (3), exporting files for the hardware (4),
simulated data (5), choosing observables (6), decoding (7), and checking the
accuracy on simulated data (8). Out of the box it runs entirely on simulated data.
For real data, set `USE_SIMULATED_DATA = False` in step 5 and point `DATA_FILE`
in step 7 at your filled-in template. To try different observables, edit the
weights in step 6 and rerun from there.

## Inputs and conventions

List all possible input frequencies, including unwanted tones. Frequencies are
in Hz and times in seconds. The field and linear observables are

```text
b(t) = sum_j [a_j*cos(2*pi*f_j*t) + b_j*sin(2*pi*f_j*t)]
F = cosine_weights @ a + sine_weights @ b
H(t) = (2*pi*coupling_hz) * b(t) * Z/2
```

Weights have shape `(N,)` for one observable or `(M,N)` for several. The bound
`amplitude_bound` applies to the joint Euclidean norm of all cosine and sine
amplitudes. Frequencies must be distinct and positive (DC is not supported);
they need not be equally spaced.

`Experiment` also takes `t2_s` (`None` removes dephasing), `min_gap_s`,
`edge_guard_s`, an optional hardware `clock_s`, calibrated `visibility` and
`readout_bias`, and per-shot `overhead_s`.

## Workflow

```python
import randomized_sensing as R

experiment = R.Experiment(
    frequencies_hz=(.8e6, 1.05e6, 1.31e6, 1.72e6),
    coupling_hz=100e3, t2_s=200e-6,
    min_gap_s=20e-9, edge_guard_s=20e-9, clock_s=2.5e-9,
)
pool = R.build_pool(experiment, duration_s=20e-6, seed=4)
plan = R.plan_acquisition(pool, n_shots=20_000, x_fraction=.2, seed=5)
R.export_bundle(plan, "outputs/randomized_experiment")
```

After acquisition, load the saved plan and your measured data, then choose
observables while decoding:

```python
plan = R.load_bundle("outputs/randomized_experiment")
counts = R.read_counts("measured_counts.csv", plan)
result = R.decode(plan, counts, method="RQS",
                  cosine_weights=(.5, -.3, 0., .4),
                  sine_weights=(.2, .4, .1, -.2))
print(result.values)
```

`R.decode(plan, counts)` returns amplitudes without observables;
`R.evaluate_observables(result.amplitudes, cosine_weights=..., sine_weights=...)`
evaluates more observables later without decoding again.

### Acquisition modes

- **Phase-locked** (default, `field_mode="phase_locked"`): every shot starts at
  the same phase of the signal, because the signal restarts each shot or the
  shots are triggered in sync with it. Repeated draws of the same sequence and
  basis are grouped into one row. Data can be integer +1 counts
  (`counts_template.csv`) or raw averaged outcomes in `[-1,1]`
  (`averages_template.csv`, `R.read_averages`, or `R.decode_averages`).
  Decode with `RQS` (default) or `RQS-ML`.
- **Free-running** (`field_mode="free_running"`): the signal runs continuously
  and is not reset. Every shot is its own row, and its **actual** start time
  must be recorded on one stable clock. Fill `plus` (0 or 1) and `start_time_s`
  in the counts CSV, or pass continuous values with
  `R.TimedOutcomes(raw_values, actual_start_times)`. Decode with `RQS`
  (default) or `TIMESTAMPED-ML`.

  ```python
  plan = R.plan_acquisition(pool, n_shots=20_000, field_mode="free_running",
                            shot_period_s=20.0375e-6, seed=5)
  ```

### Estimators

- **RQS** (both modes): moment inversion with a per-tone magnitude correction.
  For free-running data, each response is whitened in the pool frame and then
  rotated to its measured start time. There is no minimum shot count beyond one
  X and one Y shot.
- **RQS-ML** (phase-locked) / **TIMESTAMPED-ML** (free-running): optional local
  likelihood refinement starting from RQS. It is applied only with at least
  `6*N` shots (and, for TIMESTAMPED-ML, Y responses of full rank); otherwise it
  returns the RQS estimate with `ml_applied=False` and a
  `refinement_skipped_reason`.

No estimator certifies a precision or confidence interval. The notebook's
validation section checks bias without shot noise and, optionally, repeated
noisy trials.

### Pool design

The pool contains `max(8192, 64*N)` signed entries: independently drawn,
repaired random multisines and their negatives. Pool building uses FFTs when
the tones fit a common grid with the duration, and exact cell-by-cell
integration otherwise. `pool.condition` reports the covariance condition
number. New builds warn when tones are closer than `1/duration_s`, warn above
condition 20, and reject above 1000 unless `allow_ill_conditioned=True`.
`max_pulses` caps the pulse count (this changes the pool's statistics, so keep
it loose).

## Hardware files

| File | Purpose |
|---|---|
| `pulses.csv`, `intervals.csv` | Pulse times and modulation intervals |
| `acquisition.csv` | Sequence, physical X/Y basis, and repetitions per row |
| `counts_template.csv` | Copy and fill with raw physical +1 counts; plus actual start times in free-running mode |
| `averages_template.csv` | Phase-locked alternative: raw means in `[-1,1]` and repetition counts |
| `bundle.json`, `manifest.json`, `RUN.txt` | Settings, integrity hashes, and execution conventions |
| `pool.npz`, `sequences.csv` | Response pool and summaries of acquired sequences |

Read `RUN.txt` before running hardware. Save measured data separately from the
blank templates, preserving plan IDs and shot totals. Supply **raw** physical
outcomes: the decoder handles pulse-frame signs and subtracts the readout bias.

Pulses are modeled as ideal instantaneous rotations. Frequencies and Fourier
amplitudes must stay fixed during acquisition. Finite pulse widths, unrecorded
timing jitter, calibration uncertainty, and phase-reference waiting times are
not modeled.

## Layout

- `src/randomized_sensing/`: pulse pools, acquisition plans, file export/import, and decoding.
- `randomized_experiment.ipynb`: the experimental walkthrough.

The notebook writes its example files to `outputs/`, which `.gitignore` excludes.
