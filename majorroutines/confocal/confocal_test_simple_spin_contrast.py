import time
import numpy as np
import matplotlib.pyplot as plt

from utils import tool_belt as tb
from utils.constants import VirtualLaserKey
import majorroutines.targeting as targeting


def _get_pulse_duration_ns(nv_sig, vkey):
    """Use nv-specific duration if available, otherwise fall back to virtual-laser default."""
    default_ns = int(tb.get_virtual_laser_dict(vkey)["duration"])

    pulse_durations = getattr(nv_sig, "pulse_durations", None)
    if pulse_durations is None:
        return default_ns

    try:
        return int(pulse_durations.get(vkey, default_ns))
    except Exception:
        return default_ns


def plot_spin_contrast(raw_data):
    ref_counts = np.array(raw_data["ref_counts"], dtype=float)
    sig_counts = np.array(raw_data["sig_counts"], dtype=float)
    contrasts = np.array(raw_data["contrasts"], dtype=float)

    runs = np.arange(1, len(ref_counts) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(runs, ref_counts, "o-", label="ref (MW OFF)")
    plt.plot(runs, sig_counts, "s-", label="sig (MW ON)")
    plt.xlabel("Run")
    plt.ylabel("Counts")
    plt.title("Raw Counts Per Run")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(7, 5))
    plt.plot(runs, 100 * contrasts, "o-", label="Per-run contrast")
    plt.axhline(100 * raw_data["contrast_mean"], linestyle="--", label="Mean")
    plt.xlabel("Run")
    plt.ylabel("Contrast (%)")
    plt.title("Spin Contrast Per Run")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(6, 5))
    total_ref = np.sum(ref_counts)
    total_sig = np.sum(sig_counts)
    plt.bar(["ref (MW OFF)", "sig (MW ON)"], [total_ref, total_sig])
    plt.ylabel("Total Counts")
    plt.title(
        f"Summed Counts\nTotal contrast = {100 * raw_data['total_contrast']:.2f}%"
    )
    plt.tight_layout()

    plt.show()


def main(
    nv_sig,
    num_reps,
    num_runs,
    uwave_ind=0,
    uwave_freq_ghz=None,
    uwave_power_dbm=None,
    pol_ns=None,
    readout_ns=None,
    optimize_between_runs=True,
    do_plot=True,
):
    tb.reset_cfm()

    counter_server = tb.get_server_counter()
    pulsegen_server = tb.get_server_pulse_streamer()
    sig_gen = tb.get_server_sig_gen(int(uwave_ind))
    vsg = tb.get_virtual_sig_gen_dict(int(uwave_ind))

    if uwave_freq_ghz is None:
        uwave_freq_ghz = float(vsg["frequency"])
    if uwave_power_dbm is None:
        uwave_power_dbm = float(vsg["uwave_power"])

    spin_pol_vkey = VirtualLaserKey.SPIN_POL
    readout_vkey = VirtualLaserKey.SPIN_READOUT

    if pol_ns is None:
        pol_ns = _get_pulse_duration_ns(nv_sig, spin_pol_vkey)
    else:
        pol_ns = int(pol_ns)

    if readout_ns is None:
        readout_ns = _get_pulse_duration_ns(nv_sig, readout_vkey)
    else:
        readout_ns = int(readout_ns)

    seq_file = "spin_contrast_simple.py"
    seq_args = [
        int(pol_ns),
        int(readout_ns),
        int(uwave_ind),
        spin_pol_vkey,
        readout_vkey,
        None,
    ]
    seq_args_string = tb.encode_seq_args(seq_args)

    print(f"seq file = {seq_file}")
    print(f"seq args = {seq_args}")
    print(f"freq     = {uwave_freq_ghz} GHz")
    print(f"power    = {uwave_power_dbm} dBm")
    print(f"pol      = {pol_ns} ns")
    print(f"read     = {readout_ns} ns")
    print(f"pi       = {int(vsg['pi_pulse'])} ns (from config/seq)")

    ref_counts = []
    sig_counts = []
    contrasts = []

    start_time = time.time()

    try:
        for run_ind in range(num_runs):
            print(f"\nRun {run_ind + 1}/{num_runs}")

            if optimize_between_runs:
                targeting.compensate_for_drift(nv_sig, no_crash=True)

            pulsegen_server.stream_load(seq_file, seq_args_string)

            # Match the working resonance-style path:
            # reapply MW settings every run after drift compensation / stream_load
            sig_gen.set_amp(float(uwave_power_dbm))
            sig_gen.set_freq(float(uwave_freq_ghz))
            sig_gen.uwave_on()

            print(seq_file)
            print(seq_args)
            print(uwave_freq_ghz, uwave_power_dbm)

            counter_server.start_tag_stream()
            try:
                counter_server.clear_buffer()
                pulsegen_server.stream_start(int(num_reps))

                new_counts = counter_server.read_counter_modulo_gates(2, int(num_reps))
                count_arr = np.array(new_counts, dtype=np.int64)

                ref = count_arr[:, 0].sum()
                sig = count_arr[:, 1].sum()
                contrast = (ref - sig) / ref if ref > 0 else np.nan

                ref_counts.append(ref)
                sig_counts.append(sig)
                contrasts.append(contrast)

                total_ref_so_far = np.sum(ref_counts)
                total_sig_so_far = np.sum(sig_counts)
                total_contrast_so_far = (
                    (total_ref_so_far - total_sig_so_far) / total_ref_so_far
                    if total_ref_so_far > 0
                    else np.nan
                )

                print(f"ref = {ref}")
                print(f"sig = {sig}")
                print(f"contrast = {contrast:.4%}")
                print(f"cumulative contrast = {total_contrast_so_far:.4%}")

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

    finally:
        try:
            sig_gen.uwave_off()
        except Exception:
            pass
        tb.reset_cfm()

    elapsed_s = time.time() - start_time

    ref_counts = np.array(ref_counts, dtype=float)
    sig_counts = np.array(sig_counts, dtype=float)
    contrasts = np.array(contrasts, dtype=float)

    ref_mean = np.mean(ref_counts)
    sig_mean = np.mean(sig_counts)
    contrast_mean = np.mean(contrasts)
    contrast_ste = (
        np.std(contrasts, ddof=1) / np.sqrt(len(contrasts))
        if len(contrasts) > 1
        else np.nan
    )

    total_ref = np.sum(ref_counts)
    total_sig = np.sum(sig_counts)
    total_contrast = (total_ref - total_sig) / total_ref if total_ref > 0 else np.nan

    raw_data = {
        "timestamp": time.time(),
        "elapsed_s": elapsed_s,
        "nv_sig": nv_sig,
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "uwave_ind": int(uwave_ind),
        "uwave_freq_ghz": float(uwave_freq_ghz),
        "uwave_power_dbm": float(uwave_power_dbm),
        "pi_pulse_ns": int(vsg["pi_pulse"]),
        "pol_ns": int(pol_ns),
        "readout_ns": int(readout_ns),
        "ref_counts": ref_counts.tolist(),
        "sig_counts": sig_counts.tolist(),
        "contrasts": contrasts.tolist(),
        "ref_mean": float(ref_mean),
        "sig_mean": float(sig_mean),
        "contrast_mean": float(contrast_mean),
        "contrast_ste": float(contrast_ste) if not np.isnan(contrast_ste) else np.nan,
        "total_ref": float(total_ref),
        "total_sig": float(total_sig),
        "total_contrast": float(total_contrast),
    }

    print("\n=== Summary ===")
    print(f"ref mean        = {ref_mean:.1f}")
    print(f"sig mean        = {sig_mean:.1f}")
    print(f"contrast mean   = {contrast_mean:.4%}")
    print(f"contrast ste    = {contrast_ste:.4%}")
    print(f"total contrast  = {total_contrast:.4%}")
    print(f"elapsed time    = {elapsed_s:.1f} s")

    if do_plot:
        plot_spin_contrast(raw_data)

    return raw_data
