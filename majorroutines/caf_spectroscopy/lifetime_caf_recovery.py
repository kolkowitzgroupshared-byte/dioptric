import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import tool_belt as tb


def process_raw_buffer_two_gates(
    new_tags,
    new_channels,
    current_tags,
    current_channels,
    gate_open_channel,
    gate_close_channel,
    gate_counter,
):
    """
    Split alternating gate windows into:
      even gates -> readout 1
      odd gates  -> readout 2
    """
    current_tags.extend(new_tags)
    current_channels.extend(new_channels)

    current_channels_array = np.array(current_channels)

    gate_open_inds = np.nonzero(current_channels_array == gate_open_channel)[0].tolist()
    gate_close_inds = np.nonzero(current_channels_array == gate_close_channel)[
        0
    ].tolist()

    num_closed_samples = min(len(gate_open_inds), len(gate_close_inds))

    gate0_tags = []
    gate1_tags = []

    for list_ind in range(num_closed_samples):
        open_ind = gate_open_inds[list_ind]
        close_ind = gate_close_inds[list_ind]

        rep_tags = current_tags[open_ind + 1 : close_ind]
        rep_tags = np.array(rep_tags, dtype=np.int64)

        # make times relative to gate open
        rep_tags -= current_tags[open_ind]
        rep_tags = rep_tags.astype(int).tolist()

        if ((gate_counter + list_ind) % 2) == 0:
            gate0_tags.extend(rep_tags)
        else:
            gate1_tags.extend(rep_tags)

    if num_closed_samples > 0:
        leftover_start = gate_close_inds[num_closed_samples - 1]
        del current_tags[0 : leftover_start + 1]
        del current_channels[0 : leftover_start + 1]

    gate_counter += num_closed_samples
    return gate0_tags, gate1_tags, num_closed_samples, gate_counter


def _hist_from_tags(tags_ps, detect_ns, num_bins):
    detect_ps = 1000 * int(detect_ns)
    hist, _ = np.histogram(tags_ps, bins=int(num_bins), range=(0, detect_ps))

    bin_size_ns = float(detect_ns) / float(num_bins)
    bin_centers_ns = (
        np.linspace(0, float(detect_ns), int(num_bins), endpoint=False)
        + 0.5 * bin_size_ns
    )
    return hist, bin_centers_ns


def _compute_step_stats(counts_2d, detect_ns, num_reps):
    counts_2d = np.asarray(counts_2d, dtype=float)

    mean_counts = np.nanmean(counts_2d, axis=0)
    std_counts = np.nanstd(counts_2d, axis=0, ddof=1)
    n_valid = np.sum(np.isfinite(counts_2d), axis=0)

    ste_counts = np.divide(
        std_counts,
        np.sqrt(np.maximum(n_valid, 1)),
        out=np.full_like(std_counts, np.nan),
        where=n_valid > 1,
    )

    detect_s = float(detect_ns) * 1e-9
    mean_kcps = mean_counts / (float(num_reps) * detect_s) / 1e3
    ste_kcps = ste_counts / (float(num_reps) * detect_s) / 1e3

    return mean_counts, ste_counts, mean_kcps, ste_kcps


def exp_decay_with_bg(t, A, tau, C):
    return A * np.exp(-t / tau) + C


def recovery_model(t, R_inf, A, tau_meta):
    return R_inf - A * np.exp(-t / tau_meta)


def fit_lifetime(bin_centers_ns, hist, fit_start_ns=None, fit_end_ns=None):
    x = np.asarray(bin_centers_ns, dtype=float)
    y = np.asarray(hist, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if fit_start_ns is not None:
        mask &= x >= float(fit_start_ns)
    if fit_end_ns is not None:
        mask &= x <= float(fit_end_ns)

    xfit = x[mask]
    yfit = y[mask]

    if len(xfit) < 5 or np.nanmax(yfit) <= 0:
        return None

    C0 = max(np.nanmin(yfit), 0.0)
    A0 = max(np.nanmax(yfit) - C0, 1.0)
    tau0 = max((xfit[-1] - xfit[0]) / 3.0, 1.0)

    try:
        popt, pcov = curve_fit(
            exp_decay_with_bg,
            xfit,
            yfit,
            p0=[A0, tau0, C0],
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "popt": popt,
            "perr": perr,
            "xfit": xfit,
            "yfit": yfit,
        }
    except Exception as exc:
        print(f"Lifetime fit failed: {exc}")
        return None


def fit_recovery_delay(delay_ns_list, ratio_mean, ratio_ste=None):
    x = np.asarray(delay_ns_list, dtype=float)
    y = np.asarray(ratio_mean, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if ratio_ste is not None:
        sigma = np.asarray(ratio_ste, dtype=float)
        mask &= np.isfinite(sigma) & (sigma > 0)
    else:
        sigma = None

    xfit = x[mask]
    yfit = y[mask]
    sigma_fit = sigma[mask] if sigma is not None else None

    if len(xfit) < 4:
        return None

    R_inf0 = np.nanmax(yfit)
    A0 = max(R_inf0 - np.nanmin(yfit), 1e-3)
    tau0 = max((xfit[-1] - xfit[0]) / 3.0, 1.0)

    try:
        popt, pcov = curve_fit(
            recovery_model,
            xfit,
            yfit,
            p0=[R_inf0, A0, tau0],
            sigma=sigma_fit,
            absolute_sigma=(sigma_fit is not None),
            bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "popt": popt,
            "perr": perr,
            "xfit": xfit,
            "yfit": yfit,
        }
    except Exception as exc:
        print(f"Recovery fit failed: {exc}")
        return None


def main(
    sample_sig,
    num_reps,
    num_runs,
    min_recovery_delay_ns,
    max_recovery_delay_ns,
    num_steps,
    exc_ns,  # laser
    detect_ns,  # read out
    laser_buffer,
    seq_file,
    num_bins,
    filter_pos,
    laser_power=None,
    laser_vkey="SPIN_READOUT",
    do_save=True,
    fit_lifetime_start_ns=None,
    fit_lifetime_end_ns=None,
):
    tb.reset_cfm()
    if len(filter_pos) != 0:
        slider_1 = tb.get_server_slider_1()
        slider_3 = tb.get_server_slider_3()

        slider_1_pos, slider_3_pos = filter_pos
        # print(slider_1_pos, slider_3_pos)

        slider_1.set_filter(slider_1_pos)
        slider_3.set_filter(slider_3_pos)
    recovery_delay_ns_list = np.linspace(
        int(min_recovery_delay_ns),
        int(max_recovery_delay_ns),
        int(num_steps),
    )
    recovery_delay_ns_list = np.unique(np.rint(recovery_delay_ns_list).astype(int))

    timestamp = dm.get_time_stamp()

    int_counts_1 = np.full((num_runs, len(recovery_delay_ns_list)), np.nan)
    int_counts_2 = np.full((num_runs, len(recovery_delay_ns_list)), np.nan)

    hist_readout_1 = np.zeros((len(recovery_delay_ns_list), int(num_bins)), dtype=float)
    hist_readout_2 = np.zeros((len(recovery_delay_ns_list), int(num_bins)), dtype=float)

    bin_centers_ns = None

    pulsegen_server = tb.get_server_pulse_streamer()
    counter_server = tb.get_server_counter()
    tb.init_safe_stop()

    # --- NEW PROGRESS TRACKING SETUP ---
    total_steps = num_runs * len(recovery_delay_ns_list)
    current_step = 0
    print("Starting experiment...")

    start_time = time.time()

    for run_ind in range(num_runs):
        if tb.safe_stop():
            break

        for step_ind, recovery_delay_ns in enumerate(recovery_delay_ns_list):
            if tb.safe_stop():
                break

            seq_args = [
                int(recovery_delay_ns),
                int(exc_ns),
                int(detect_ns),
                int(laser_buffer),
                laser_vkey,
                laser_power,
            ]
            seq_args_string = tb.encode_seq_args(seq_args)
            ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)

            # We can keep this one so you see the sequence period once at the very start
            if run_ind == 0 and step_ind == 0:
                print(f"Sequence period: {ret_vals[0]} ns\n")

            counter_server.start_tag_stream()
            try:
                pulsegen_server.stream_start(int(num_reps))

                channel_mapping = counter_server.get_channel_mapping()
                gate_open_channel = channel_mapping[1]
                gate_close_channel = channel_mapping[2]

                current_tags = []
                current_channels = []

                gate_counter = 0
                num_processed_gates = 0
                target_num_gates = 2 * int(num_reps)

                readout1_tags = []
                readout2_tags = []

                while num_processed_gates < target_num_gates:
                    if tb.safe_stop():
                        break

                    new_tags, new_channels = counter_server.read_tag_stream()
                    new_tags = np.array(new_tags, dtype=np.int64)

                    g0_tags, g1_tags, num_new_gates, gate_counter = (
                        process_raw_buffer_two_gates(
                            new_tags=new_tags,
                            new_channels=new_channels,
                            current_tags=current_tags,
                            current_channels=current_channels,
                            gate_open_channel=gate_open_channel,
                            gate_close_channel=gate_close_channel,
                            gate_counter=gate_counter,
                        )
                    )

                    readout1_tags.extend(g0_tags)
                    readout2_tags.extend(g1_tags)
                    num_processed_gates += num_new_gates

                step_hist_1, bin_centers_ns = _hist_from_tags(
                    readout1_tags, detect_ns, num_bins
                )
                step_hist_2, _ = _hist_from_tags(readout2_tags, detect_ns, num_bins)

                hist_readout_1[step_ind] += step_hist_1
                hist_readout_2[step_ind] += step_hist_2

                int_counts_1[run_ind, step_ind] = np.sum(step_hist_1)
                int_counts_2[run_ind, step_ind] = np.sum(step_hist_2)

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            # --- NEW CLEAN PROGRESS BAR & ETA ---
            current_step += 1
            elapsed = time.time() - start_time
            time_per_step = elapsed / current_step
            eta_seconds = time_per_step * (total_steps - current_step)

            # Format ETA into mm:ss
            mins, secs = divmod(int(eta_seconds), 60)
            eta_str = f"{mins:02d}:{secs:02d}"

            # \r forces the cursor to the start of the line, end="" prevents a new line, flush=True forces the terminal to update
            print(
                f"\rProgress: [{current_step}/{total_steps}] steps | ETA: {eta_str}   ",
                end="",
                flush=True,
            )

    # Print a new line at the very end so the final summary text doesn't overwrite our completed progress bar
    print(
        f"\n\nExperiment finished! Total elapsed time: {time.time() - start_time:.2f} s"
    )

    # --- Data Processing (Kept identical so your fits still save correctly) ---
    mean_counts_1, ste_counts_1, mean_kcps_1, ste_kcps_1 = _compute_step_stats(
        int_counts_1, detect_ns, num_reps
    )
    mean_counts_2, ste_counts_2, mean_kcps_2, ste_kcps_2 = _compute_step_stats(
        int_counts_2, detect_ns, num_reps
    )

    ratio_runs = np.divide(
        int_counts_2,
        int_counts_1,
        out=np.full_like(int_counts_2, np.nan),
        where=int_counts_1 > 0,
    )
    ratio_mean = np.nanmean(ratio_runs, axis=0)
    ratio_ste = np.nanstd(ratio_runs, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(np.isfinite(ratio_runs), axis=0), 1)
    )

    best_ind = int(np.nanargmax(recovery_delay_ns_list))

    lifetime_fit_1 = fit_lifetime(
        bin_centers_ns,
        hist_readout_1[best_ind],
        fit_start_ns=fit_lifetime_start_ns,
        fit_end_ns=fit_lifetime_end_ns,
    )
    lifetime_fit_2 = fit_lifetime(
        bin_centers_ns,
        hist_readout_2[best_ind],
        fit_start_ns=fit_lifetime_start_ns,
        fit_end_ns=fit_lifetime_end_ns,
    )
    recovery_fit = fit_recovery_delay(
        recovery_delay_ns_list,
        ratio_mean,
        ratio_ste=ratio_ste,
    )

    # --- Simplified Static Plotting at the End ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(
        recovery_delay_ns_list,
        mean_counts_1,
        yerr=ste_counts_1,
        fmt="ro-",
        label="Readout 1",
    )
    ax1.errorbar(
        recovery_delay_ns_list,
        mean_counts_2,
        yerr=ste_counts_2,
        fmt="bo-",
        label="Readout 2",
    )
    ax1.set_ylabel("Integrated Counts")
    ax1.set_title("Double Lifetime Recovery Summary")
    ax1.legend()

    ax2.errorbar(recovery_delay_ns_list, ratio_mean, yerr=ratio_ste, fmt="ko-")

    # Optional: Plot the recovery fit line if it succeeded
    if recovery_fit is not None:
        xfine = np.linspace(
            min(recovery_delay_ns_list), max(recovery_delay_ns_list), 200
        )
        ax2.plot(xfine, recovery_model(xfine, *recovery_fit["popt"]), "k--", alpha=0.5)

    ax2.set_xlabel("Dark Recovery Delay (ns)")
    ax2.set_ylabel("Ratio (Readout 2 / Readout 1)")

    plt.tight_layout()
    plt.show()

    proc_data = {
        "recovery_delay_ns_list": recovery_delay_ns_list.tolist(),
        "bin_centers_ns": bin_centers_ns.tolist()
        if bin_centers_ns is not None
        else None,
        "hist_readout_1": hist_readout_1.tolist(),
        "hist_readout_2": hist_readout_2.tolist(),
        "mean_counts_1": mean_counts_1.tolist(),
        "mean_counts_2": mean_counts_2.tolist(),
        "ste_counts_1": ste_counts_1.tolist(),
        "ste_counts_2": ste_counts_2.tolist(),
        "mean_kcps_1": mean_kcps_1.tolist(),
        "mean_kcps_2": mean_kcps_2.tolist(),
        "ste_kcps_1": ste_kcps_1.tolist(),
        "ste_kcps_2": ste_kcps_2.tolist(),
        "ratio_mean": ratio_mean.tolist(),
        "ratio_ste": ratio_ste.tolist(),
        "lifetime_fit_1": None
        if lifetime_fit_1 is None
        else {
            "A": float(lifetime_fit_1["popt"][0]),
            "tau_ns": float(lifetime_fit_1["popt"][1]),
            "C": float(lifetime_fit_1["popt"][2]),
            "A_err": float(lifetime_fit_1["perr"][0]),
            "tau_ns_err": float(lifetime_fit_1["perr"][1]),
            "C_err": float(lifetime_fit_1["perr"][2]),
        },
        "lifetime_fit_2": None
        if lifetime_fit_2 is None
        else {
            "A": float(lifetime_fit_2["popt"][0]),
            "tau_ns": float(lifetime_fit_2["popt"][1]),
            "C": float(lifetime_fit_2["popt"][2]),
            "A_err": float(lifetime_fit_2["perr"][0]),
            "tau_ns_err": float(lifetime_fit_2["perr"][1]),
            "C_err": float(lifetime_fit_2["perr"][2]),
        },
        "recovery_fit": None
        if recovery_fit is None
        else {
            "R_inf": float(recovery_fit["popt"][0]),
            "A": float(recovery_fit["popt"][1]),
            "tau_meta_ns": float(recovery_fit["popt"][2]),
            "R_inf_err": float(recovery_fit["perr"][0]),
            "A_err": float(recovery_fit["perr"][1]),
            "tau_meta_ns_err": float(recovery_fit["perr"][2]),
        },
    }

    raw_data = {
        "timestamp": timestamp,
        "sample_name": getattr(sample_sig, "name", "sample"),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "exc_ns": int(exc_ns),
        "detect_ns": int(detect_ns),
        "laser_buffer": int(laser_buffer),
        "num_bins": int(num_bins),
        "laser_vkey": str(laser_vkey),
        "laser_power": laser_power,
        "recovery_delay_ns_list": recovery_delay_ns_list.tolist(),
        "int_counts_1": int_counts_1.tolist(),
        "int_counts_2": int_counts_2.tolist(),
    }

    if do_save:
        file_path = dm.get_file_path(
            __file__,
            timestamp,
            getattr(sample_sig, "name", "sample"),
        )
        dm.save_raw_data(raw_data, file_path)
        dm.save_raw_data(proc_data, file_path + "_proc")
        # if fig is not None:
        #     dm.save_figure(fig, file_path)
        print(f"Saved data to {file_path}")

    if proc_data["recovery_fit"] is not None:
        print(
            f"Recovered metastable time = "
            f"{proc_data['recovery_fit']['tau_meta_ns']:.3f} ± "
            f"{proc_data['recovery_fit']['tau_meta_ns_err']:.3f} ns"
        )

    tb.reset_cfm()
    return raw_data, proc_data


# if __name__ == "__main__":

#     class Dummy:
#         name = "caf_test"

#     sample_sig = Dummy()

#     # simulation test
#     raw_data, proc_data = main(
#         sample_sig=sample_sig,
#         num_reps=2000,
#         num_runs=5,
#         min_recovery_delay_ns=0,
#         max_recovery_delay_ns=30000,
#         num_steps=31,
#         exc_ns=1000,
#         detect_ns=3000,
#         seq_file="lifetime_caf_recovery.py",
#         num_bins=150,
#         laser_power=None,
#         laser_vkey="SPIN_READOUT",
#         do_save=False,
#         fit_lifetime_start_ns=0,
#         fit_lifetime_end_ns=2500,
#     )

# for real experiment:
# raw_data, proc_data = main(
#     sample_sig=sample_sig,
#     num_reps=5000,
#     num_runs=10,
#     min_recovery_delay_ns=0,
#     max_recovery_delay_ns=50000,
#     num_steps=41,
#     exc_ns=1000,
#     detect_ns=4000,
#     num_bins=200,
#     laser_power=None,
#     laser_vkey="SPIN_READOUT",
#     do_plot=True,
#     do_save=True,
#     simulate_only=False,
#     fit_lifetime_start_ns=0,
#     fit_lifetime_end_ns=3500,
# )
