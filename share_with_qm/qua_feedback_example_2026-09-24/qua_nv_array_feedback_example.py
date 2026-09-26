"""
Minimal QUA-side example of our multi-NV camera-feedback pattern.

The essential point is that the target mask is supplied by the host at runtime,
while x/y frequencies and per-NV pulse parameters are currently static vectors
embedded in the QUA program.
"""

from qm import qua


def build_program(x_freq_hz, y_freq_hz, pol_duration_cc, pol_amp, num_reps=1):
    num_nvs = len(x_freq_hz)

    with qua.program() as seq:
        # Runtime data supplied by the host.
        charge_pol_incomplete = qua.declare_input_stream(
            bool, name="charge_pol_incomplete"
        )
        target_mask = qua.declare_input_stream(
            bool, name="target_mask", size=num_nvs
        )

        # Variables used by the compiled per-NV loop.
        x_freq = qua.declare(int)
        y_freq = qua.declare(int)
        duration = qua.declare(int)
        amplitude = qua.declare(qua.fixed)
        target = qua.declare(bool)

        def polarize_selected_nvs():
            # Blocks until the next N_NV-element mask is available.
            qua.advance_input_stream(target_mask)

            # This mirrors our present implementation.
            # These four parameter vectors are compiled into the program.
            with qua.for_each_(
                (x_freq, y_freq, duration, amplitude, target),
                (
                    x_freq_hz,
                    y_freq_hz,
                    pol_duration_cc,
                    pol_amp,
                    target_mask,
                ),
            ):
                with qua.if_(target):
                    qua.update_frequency("green_aod_x", x_freq)
                    qua.update_frequency("green_aod_y", y_freq)
                    qua.wait(100, "green_laser")
                    qua.play(
                        "charge_pol" * qua.amp(amplitude),
                        "green_laser",
                        duration=duration,
                    )

        def camera_charge_readout():
            qua.align()
            qua.play("charge_readout", "yellow_laser")
            qua.play("on", "camera_trigger")
            # Camera returns a hardware-ready/acquisition-complete trigger.
            qua.wait_for_trigger("camera_trigger")

        rep = qua.declare(int)
        with qua.for_(rep, 0, rep < num_reps, rep + 1):
            # Host tells QUA whether another selective preparation pass is needed.
            qua.advance_input_stream(charge_pol_incomplete)

            with qua.while_(charge_pol_incomplete):
                polarize_selected_nvs()
                camera_charge_readout()

                # Host processes the new frame and supplies the next status/mask.
                qua.advance_input_stream(charge_pol_incomplete)

            # Spin-control sequence would follow here.
            # e.g. microwave pi/2 - evolution - pi - evolution - pi/2 - SCC/readout

        qua.pause()

    return seq
