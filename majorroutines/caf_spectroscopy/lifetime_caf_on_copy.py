import time

import matplotlib.pyplot as plt
import numpy as np

import utils.tool_belt as tool_belt


def main(readout_total_s, num_reps, laser_power=0.1e-3):
    tool_belt.reset_cfm()
    pulsegen_server = tool_belt.get_server_pulse_streamer()
    counter_server = tool_belt.get_server_counter()

    # --- THE TRICK: Sub-dividing the 10s into 100ms chunks ---
    chunk_ns = int(0.1e9)  # 100ms chunks
    chunks_per_rep = int((readout_total_s * 1e9) / chunk_ns)

    # We load a 100ms sequence and just run it for many repetitions
    seq_args = [0, chunk_ns, "SPIN_READOUT", laser_power]
    pulsegen_server.stream_load(
        "simple_readout_test.py", tool_belt.encode_seq_args(seq_args)
    )

    # Setup Plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot([], [], "r-o", markersize=4)
    ax.set_title(f"10s Repetition Readout (Total chunks: {chunks_per_rep * num_reps})")
    ax.set_xlabel("Repetition #")
    ax.set_ylabel("Total Counts (per 10s)")

    reps_data = []
    current_rep_accumulator = []

    counter_server.start_tag_stream()
    # Run essentially "forever" until we manually stop or reps finish
    pulsegen_server.stream_start(-1)
    tool_belt.init_safe_stop()

    print(f"Acquiring {num_reps} reps of {readout_total_s}s each...")

    while len(reps_data) < num_reps:
        if tool_belt.safe_stop():
            break

        # Read the 100ms chunks
        new = counter_server.read_counter_simple()

        if new is not None and len(new) > 0:
            new_data = np.array(new).flatten()
            current_rep_accumulator.extend(new_data)

            # Check if we have enough chunks to complete one 10s rep
            while len(current_rep_accumulator) >= chunks_per_rep:
                # Sum the first 100 chunks to make one 10s data point
                one_full_rep = sum(current_rep_accumulator[:chunks_per_rep])
                reps_data.append(one_full_rep)

                # Print progress
                print(f"Completed Rep {len(reps_data)}: {one_full_rep} counts")

                # Remove the used chunks
                current_rep_accumulator = current_rep_accumulator[chunks_per_rep:]

            # Update Plot
            if reps_data:
                line.set_data(range(1, len(reps_data) + 1), reps_data)
                ax.set_xlim(0.5, num_reps + 0.5)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

        time.sleep(0.2)  # Breathe

    # Stop everything
    pulsegen_server.stop()

    plt.ioff()
    plt.show()

    print("hello")


if __name__ == "__main__":
    # 10 seconds per rep, 5 reps total
    main(readout_total_s=10.0, num_reps=5, laser_power=0.1e-3)
