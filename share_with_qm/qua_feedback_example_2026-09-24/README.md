# QM / QUA multi-NV feedback example

This folder is a compact, discussion-oriented extract of our current widefield NV-control architecture.
It is intentionally independent of the rest of our lab framework so that the key QUA/host interaction is easy to review.

## Experiment context

We control hundreds to thousands of individually resolved NV centers.
Per-NV optical addressing is performed with AODs; charge-state readout is camera based.
After each camera frame, the host PC estimates the charge state of every NV and builds a Boolean target mask.
Only NVs that are not in the desired charge state are re-addressed.

## Current control flow

1. Build and compile the QUA program.
2. Start the compiled job and arm the camera.
3. QUA performs charge readout and triggers the camera.
4. The host reads the image and estimates the state of every NV.
5. The host sends:
   - a Boolean indicating whether charge preparation is still incomplete;
   - a Boolean target mask of length N_NV.
6. QUA consumes the input streams and selectively polarizes only targeted NVs.
7. Readout/feedback repeats until complete (or a host-side maximum-attempt limit is reached).

## Important implementation detail

The target mask is updated at runtime, but the per-NV x/y AOD frequencies, pulse durations,
and pulse amplitudes are currently embedded in the QUA program and traversed with qua.for_each_().
In our scaling test the program compiled for 4095 NVs but failed at 4096 NVs.
Our intended scale is roughly 1,000-4,000 NVs initially and potentially ~10,000 longer term.

## Current software

QM Python package on the experimental computer: 1.1.7.
Our present server wrapper calls running_job.insert_input_stream(...).
QM's current API documentation recommends push_to_input_stream(...); we would appreciate guidance
on the preferred API/architecture for OPX1000.

## Files

- qua_nv_array_feedback_example.py: simplified QUA-side implementation matching our current pattern.
- host_feedback_example.py: simplified host-side camera processing / mask-update loop.
- QUESTIONS_FOR_QM.md: specific questions on scaling, runtime updates, camera feedback, LF-FEM and MW-FEM.

## Mapping to our lab code

The example is distilled from:
- servers/timing/sequencelibrary/QM_opx/seq_utils.py
- majorroutines/widefield/base_routine.py
- servers/mixed/QM_opx.py
- servers/timing/sequencelibrary/QM_opx/camera/charge_state_conditional_init.py

This is not a standalone hardware configuration; element and pulse names are representative.
