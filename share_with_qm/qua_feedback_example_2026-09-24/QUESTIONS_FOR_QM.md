# Questions for Quantum Machines

## 1. Scaling beyond the current qua.for_each_ vector limit

Our current implementation embeds x/y AOD frequencies, pulse durations and amplitudes in
qua.for_each_() vectors. We observed successful compilation at 4095 NVs and failure at 4096.
Is this expected from the for_each_/array resource limit? What pattern do you recommend for
~1,000-4,000 NVs today and potentially ~10,000?

## 2. Runtime update of large per-NV arrays

Can a running OPX1000 program efficiently receive and use large arrays for:
- target mask (bool);
- x frequency (int);
- y frequency (int);
- duration (int);
- amplitude (fixed);
without recompiling the QUA program?

Would you recommend input streams, declared QUA arrays filled from streams, another API,
or a different loop structure?

## 3. Host-to-running-job latency / throughput

For every camera frame we may send a 1,000-4,000 element Boolean target mask.
What host-to-OPX latency and sustained update rate should we expect?
Is one vector push preferred over many scalar updates?
Are there practical queue-depth or memory limits we should design around?

## 4. Recommended camera-feedback architecture

Our current pattern is:
QUA triggers camera -> camera acquisition-complete hardware trigger -> host reads/processes frame
-> host pushes new mask/status -> QUA continues.

Is advance_input_stream() the preferred blocking primitive for this?
Would pause()/resume(), IO variables, or another OPX1000 mechanism be better for low-latency,
reliable repeated feedback?

## 5. Updating AOD coordinates after drift correction

Between runs we update NV coordinates for sample drift.
Can the x/y frequency arrays used by the AODs be replaced in a running/precompiled program,
so that a coordinate update does not require a new compile?

## 6. LF-FEM multitone AOD control

We use four RF channels for green/red AOD x/y control around 75-110 MHz and sometimes want
multiple simultaneous tones, fast frequency hops/chirps, and per-tone amplitude control.
What is the recommended LF-FEM implementation for this, and how many independently controlled
tones can we generate per output while retaining deterministic timing?

## 7. MW-FEM multitone / phase control

For NV spin control we need pi/2 and pi pulses as short as ~24-48 ns and phase-controlled sequences.
For multiple simultaneous microwave tones on one MW-FEM output, can frequency, amplitude and phase
be independently controlled per tone and updated during a running sequence?
What are the relevant limits on number of simultaneous tones and instantaneous bandwidth?

## 8. API modernization

Our installed QM package is 1.1.7 and our wrapper currently calls insert_input_stream().
Current documentation marks that method deprecated in favor of push_to_input_stream().
Should we migrate directly, and are there other QOP/QUA changes you recommend before moving this
architecture to an OPX1000?
