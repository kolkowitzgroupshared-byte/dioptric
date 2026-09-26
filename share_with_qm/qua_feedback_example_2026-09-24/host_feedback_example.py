"""
Simplified host-side logic corresponding to our current widefield base routine.

In the real experiment camera.read() returns the frame, the host integrates counts
around each NV, applies threshold/MLE charge-state estimation, and creates a Boolean
mask selecting only NVs that still need charge polarization.
"""


def build_target_mask(states):
    # Convention used in our current code:
    # state 0 (or unknown) -> target this NV again
    return [state is None or state == 0 for state in states]


def run_charge_feedback(job, camera, estimate_states, num_nvs, max_attempts=10):
    states = None

    for attempt in range(max_attempts + 1):
        target_mask = (
            [True] * num_nvs if states is None else build_target_mask(states)
        )

        complete = (True not in target_mask) or (attempt == max_attempts)

        # Current dioptric wrapper uses:
        #   running_job.insert_input_stream(name, value)
        # Current QM docs recommend push_to_input_stream().
        job.push_to_input_stream("charge_pol_incomplete", not complete)

        if complete:
            break

        # Send the full per-NV selection mask in one host call.
        job.push_to_input_stream("target_mask", target_mask)

        # QUA performs selective polarization, triggers a charge-readout frame,
        # and waits for the camera acquisition-complete signal.
        frame = camera.read()

        # Host-side image processing / thresholding or MLE.
        states = estimate_states(frame)

    return states


def desired_extension(job, x_freq_hz, y_freq_hz, durations, amplitudes, mask):
    """
    What we would ideally like to do without recompiling the program:
      - update target mask;
      - optionally update x/y AOD frequency arrays after drift correction;
      - update per-NV pulse duration/amplitude arrays;
      - continue the same running QUA program.

    We would like QM's recommendation for the most scalable OPX1000 pattern
    for N_NV ~ 1,000-4,000 now and potentially ~10,000.
    """
    raise NotImplementedError("Architecture question for QM")
