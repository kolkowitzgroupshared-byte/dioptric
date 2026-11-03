SOURCE ORIGINAL RQS BUNDLE: C:\Users\saroj\Github\dioptric\RQS_toolkit\RQS_toolkit\outputs\rqs_phase_locked
UNMODIFIED: pulses.csv, sequences.csv, acquisition.csv, pool.npz, bundle.json
ACQUISITION ROWS: 9074
PHYSICAL SEQUENCES: 7492
TOTAL SHOTS: 20000
X / Y SHOTS: 2000 / 18000
SENSING DURATION: 100000 ns
PI / PI2 DURATION: 88 / 44 ns
CLOCK: 4 ns, MIN FREE WAIT: 16 ns
FIRST-PI2 START TO FINAL-PI2 END: 100088 ns

TIMING CONVENTIONS:
  sensing t=0 is immediately after initial pi/2 (not at its beginning)
  pi center c (sensing-relative) => physical pi starts at c - pi_ns/2
  in time relative to first pi/2 start, pi starts at pi2_ns+c-pi_ns/2
  waits_ns and waits_cc contain one wait after initial pi/2 and one
  after EVERY refocusing pi before final analysis pi/2
  final pi/2 analysis basis comes from each acquisition row (X/Y)
  opx_full_microwave_sequence.csv explicitly lists INITIAL/FINAL pi/2
  and every refocusing pi for every row_id, in physical gate order.
  Nominal axes: PREPARE +Y, REFOCUS +X, ANALYZE X: -Y, Y: +X.
  Calibrate axes/signs against the lab's actual microwave convention.

IMPORTANT: outcome_sign_DECODER_ONLY is not a microwave command.
  Preserve RAW physical X/Y measurements keyed to original row_id.
  The ORIGINAL RQS decoder applies virtual-sign/parity corrections.
  Full gate schedule is per acquisition row, NOT per individual shot.
  Repeat each row's schedule by the specified shots count.
  Do not modify the original bundle or its saved response vectors.
  Finite-pulse evolution is NOT corrected in the original ideal response.
  Verify QUA macro durations, X/Y phases, trigger/latency and SCC timing.
