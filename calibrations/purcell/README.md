# Purcell calibration hierarchy

This folder contains the **offline-safe active calibration state** for the Purcell experiment computer.

## Current runtime files

`current/` is the only calibration location that production Purcell code should load directly.

- `active_nv_coords.npz` — active NV coordinates and SLM weights
- `slm_fourier.h5` — active SLM Fourier calibration
- `nuvu_to_thorcam_slm.npz` — Nuvu-to-ThorCam SLM registration
- `dmd_zero_order.npz` — active DMD zero-order calibration
- `dmd_triangle_affine.npz` — active DMD triangle affine calibration
- `nuvu_to_thorcam_dmd.npz` — Nuvu-to-ThorCam DMD registration
- `dmd_nv_chain.npz` — final Nuvu-to-DMD chain used by the DMD server

These files are deliberately kept in the local repository so the experiment can start even if `G:\nvdata` is temporarily unavailable.

## Calibration history

New calibration results should be written first through `utils.data_manager` into the standard hierarchy under:

`G:\nvdata\pc_Purcell\branch_<branch>\<source>\YYYY_MM\...`

Only after validation should a result be promoted into `current/`.

The G drive is the canonical provenance/history store. The local `current/` directory is the canonical runtime snapshot.

## Promotion rule

1. Generate calibration and save timestamped result to G.
2. Validate the result.
3. Back up the existing local current file.
4. Copy the validated result into the corresponding `current/` filename.
5. Verify SHA-256 after the copy.
6. Runtime code continues to load only the stable local filename.

Do not silently overwrite current calibration files during exploratory calibration runs.
