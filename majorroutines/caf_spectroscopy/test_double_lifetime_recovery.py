
# double_lifetime_recovery_main.py
#
# Double-pulse lifetime recovery experiment:
#
#   pulse 1 -> lifetime readout 1 -> dark delay -> pulse 2 -> lifetime readout 2
#
# For every recovery delay:
#   1. collect photon arrival times in readout 1 and readout 2
#   2. build lifetime histograms
#   3. integrate counts
#   4. compute readout2 / readout1
#
# Fits:
#   lifetime histograms: A exp(-t / tau) + C
#   recovery ratio:      R_inf - A exp(-t / tau_recovery)

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from utils import data_manager as dm
from utils import kplotlib as kpl
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
    Split alternating APD gates into two groups.

    even gates -> readout 1
    odd gates  -> readout 2

    Photon times are returned relative to their own APD gate opening.

    Assumes the tagger timestamps are in ps.
    """

    current_tags.extend(new_tags)
    current_channels.extend(new_channels)

    current_channels_array = np.array(current_channels)

    gate_open_inds = np.nonzero(current_channels_array == gate_open_channel)[0].tolist()
    gate_close_inds = np.nonzero(current_channels_array == gate_close_channel)[0].tolist()

    num_closed_gates = min(len(gate_open_inds), len(gate_close_inds))

    readout1_tags = []
    readout2_tags = []

    for local_gate_ind in range(num_closed_gates):
        open_ind = gate_open_inds[local_gate_ind]
        close_ind = gate_close_inds[local_gate_ind]

        if close_ind <= open_ind:
            continue

        gate_tags = current_tags[open_ind + 1 : close_ind]
        gate_tags = np.array(gate_tags, dtype=np.int64)

        # Times relative to APD gate opening
        gate_tags -= current_tags[open_ind]
        gate_tags = gate_tags.astype(int).tolist()

        global_gate_ind = gate_counter + local_gate_ind

        if global_gate_ind % 2 == 0:
            readout1_tags.extend(gate_tags)
        else:
            readout2_tags.extend(gate_tags)

    if num_closed_gates > 0:
        leftover_start = gate_close_inds[num_closed_gates - 1]
        del current_tags[0 : leftover_start + 1]
        del current_channels[0 : leftover_start + 1]

    gate_counter += num_closed_gates

    return readout1_tags, readout2_tags, num_closed_gates, gate_counter


def hist_from_tags(tags_ps, detect_ns, num_bins):
    """
    Build histogram from photon arrival times.

    tags_ps: photon times relative to gate opening, in ps
    detect_ns: APD gate width, in ns
    """

    detect_ps = int(detect_ns) * 1000

    hist, edges_ps = np.histogram(
        tags_ps,
        bins=int(num_bins),
        range=(0, detect_ps),
    )

    bin_centers_ps = 0.5 * (edges_ps[:-1] + edges_ps[1:])
    bin_centers_ns = bin_centers_ps / 1000.0

    return hist.astype(float), bin_centers_ns


def exp_decay_with_bg(t_ns, A, tau_ns, C):
    return A * np.exp(-t_ns / tau_ns) + C


def recovery_model(t_ns, R_inf, A, tau_recovery_ns):
    return R_inf - A * np.exp(-t_ns / tau_recovery_ns)


def fit_lifetime(
    bin_centers_ns,
    hist,
    fit_start_ns=None,
    fit_end_ns=None,
):
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

    C0 = max(float(np.nanmin(yfit)), 0.0)
    A0 = max(float(np.nanmax(yfit) - C0), 1.0)
    tau0 = max((float(xfit[-1]) - float(xfit[0])) / 3.0, 1.0)

    sigma = np.sqrt(np.maximum(yfit, 1.0))

    try:
        popt, pcov = curve_fit(
            exp_decay_with_bg,
            xfit,
            yfit,
            p0=[A0, tau0, C0],
            sigma=sigma,
            absolute_sigma=False,
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


def fit_recovery(
    effective_delay_ns_list,
    ratio_mean,
    ratio_ste=None,
):
    x = np.asarray(effective_delay_ns_list, dtype=float)
    y = np.asarray(ratio_mean, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    sigma_fit = None
    if ratio_ste is not None:
        sigma = np.asarray(ratio_ste, dtype=float)
        mask &= np.isfinite(sigma) & (sigma > 0)
        sigma_fit = sigma[mask]

    xfit = x[mask]
    yfit = y[mask]

    if len(xfit) < 4:
        return None

    R_inf0 = float(np.nanmax(yfit))
    A0 = max(R_inf0 - float(np.nanmin(yfit)), 1e-3)
    tau0 = max((float(xfit[-1]) - float(xfit[0])) / 3.0, 1.0)

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


def integrate_histogram(
    hist,
    bin_centers_ns,
    integrate_start_ns=None,
    integrate_end_ns=None,
    subtract_tail_bg=False,
    tail_fraction=0.25,
):
    """
    Integrate histogram counts over a selected time window.

    If subtract_tail_bg=True:
      estimate constant background from the last tail_fraction of the full trace
      and subtract it bin-by-bin before integration.
    """

    hist = np.asarray(hist, dtype=float)
    t = np.asarray(bin_centers_ns, dtype=float)

    y = hist.copy()

    if subtract_tail_bg:
        n_tail = max(1, int(len(y) * float(tail_fraction)))
        bg = np.nanmean(y[-n_tail:])
        y = y - bg
        y[y < 0] = 0.0

    mask = np.isfinite(t) & np.isfinite(y)

    if integrate_start_ns is not None:
        mask &= t >= float(integrate_start_ns)

    if integrate_end_ns is not None:
        mask &= t <= float(integrate_end_ns)

    return float(np.nansum(y[mask]))


def main(
    sample_sig,
    num_reps,
    num_runs,
    min_recovery_delay_ns,
    max_recovery_delay_ns,
    num_steps,
    exc_ns,
    readout_delay_ns,
    detect_ns,
    num_bins=100,
    laser_vkey="SPIN_READOUT",
    laser_power=None,
    integrate_start_ns=0,
    integrate_end_ns=None,
    subtract_tail_bg=False,
    fit_lifetime_start_ns=0,
    fit_lifetime_end_ns=None,
    randomize_delay_order=True,
    do_plot=True,
    do_save=True,
):
    """
    Double-pulse recovery experiment.

    Important timing definition:

        recovery_delay_ns = extra dark time after readout 1
        effective_delay_ns = readout_delay_ns + detect_ns + recovery_delay_ns

    The effective delay is the time from the end of excitation pulse 1
    to the start of excitation pulse 2.
    """

    tb.reset_cfm()
    kpl.init_kplotlib()

    timestamp = dm.get_time_stamp()

    recovery_delay_ns_list = np.linspace(
        int(min_recovery_delay_ns),
        int(max_recovery_delay_ns),
        int(num_steps),
    )
    recovery_delay_ns_list = np.unique(
        np.rint(recovery_delay_ns_list).astype(int)
    )

    num_steps_actual = len(recovery_delay_ns_list)

    effective_delay_ns_list = (
        recovery_delay_ns_list + int(readout_delay_ns) + int(detect_ns)
    )

    hist_readout_1 = np.zeros((num_steps_actual, int(num_bins)), dtype=float)
    hist_readout_2 = np.zeros((num_steps_actual, int(num_bins)), dtype=float)

    int_counts_1 = np.full((int(num_runs), num_steps_actual), np.nan)
    int_counts_2 = np.full((int(num_runs), num_steps_actual), np.nan)

    bin_centers_ns = None

    seq_file = "double_lifetime_recovery.py"

    pulsegen_server = tb.get_server_pulse_streamer()
    counter_server = tb.get_server_counter()

    if do_plot:
        fig, axes = plt.subplots(3, 1, figsize=(9, 12), constrained_layout=True)

        ax_counts = axes[0]
        ax_ratio = axes[1]
        ax_lifetime = axes[2]

        ax_counts.set_title("Double lifetime recovery")
        ax_counts.set_xlabel("Effective delay: end pulse 1 to start pulse 2 (ns)")
        ax_counts.set_ylabel("Integrated counts")

        ax_ratio.set_xlabel("Effective delay: end pulse 1 to start pulse 2 (ns)")
        ax_ratio.set_ylabel("Readout 2 / Readout 1")

        ax_lifetime.set_xlabel("Photon arrival time after APD gate opens (ns)")
        ax_lifetime.set_ylabel("Counts")

        line_counts_1, = ax_counts.plot([], [], "o-", label="readout 1")
        line_counts_2, = ax_counts.plot([], [], "o-", label="readout 2")
        line_ratio, = ax_ratio.plot([], [], "o-", label="ratio")

        ax_counts.legend()
        ax_ratio.legend()

    else:
        fig = None
        ax_counts = None
        ax_ratio = None
        ax_lifetime = None
        line_counts_1 = None
        line_counts_2 = None
        line_ratio = None

    tb.init_safe_stop()
    start_time = time.time()

    for run_ind in range(int(num_runs)):
        if tb.safe_stop():
            break

        print(f"Run {run_ind + 1}/{num_runs}")

        step_order = np.arange(num_steps_actual)

        if randomize_delay_order:
            np.random.shuffle(step_order)

        for order_ind, step_ind in enumerate(step_order):
            if tb.safe_stop():
                break

            recovery_delay_ns = int(recovery_delay_ns_list[step_ind])
            effective_delay_ns = int(effective_delay_ns_list[step_ind])

            print(
                f"  step {order_ind + 1}/{num_steps_actual} | "
                f"recovery_delay = {recovery_delay_ns} ns | "
                f"effective_delay = {effective_delay_ns} ns"
            )

            seq_args = [
                int(recovery_delay_ns),
                int(exc_ns),
                int(readout_delay_ns),
                int(detect_ns),
                laser_vkey,
                laser_power,
            ]
            seq_args_string = tb.encode_seq_args(seq_args)

            ret_vals = pulsegen_server.stream_load(seq_file, seq_args_string)

            if run_ind == 0 and order_ind == 0:
                print(f"  Sequence period: {ret_vals[0]} ns")

            current_tags = []
            current_channels = []

            gate_counter = 0
            num_processed_gates = 0
            target_num_gates = 2 * int(num_reps)

            readout1_tags = []
            readout2_tags = []

            counter_server.start_tag_stream()

            try:
                pulsegen_server.stream_start(int(num_reps))

                channel_mapping = counter_server.get_channel_mapping()
                gate_open_channel = channel_mapping[1]
                gate_close_channel = channel_mapping[2]

                while num_processed_gates < target_num_gates:
                    if tb.safe_stop():
                        break

                    new_tags, new_channels = counter_server.read_tag_stream()

                    if len(new_tags) == 0:
                        continue

                    new_tags = np.array(new_tags, dtype=np.int64)

                    g1_tags, g2_tags, num_new_gates, gate_counter = (
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

                    readout1_tags.extend(g1_tags)
                    readout2_tags.extend(g2_tags)
                    num_processed_gates += num_new_gates

            finally:
                try:
                    counter_server.stop_tag_stream()
                except Exception:
                    pass

            step_hist_1, bin_centers_ns = hist_from_tags(
                readout1_tags,
                detect_ns=detect_ns,
                num_bins=num_bins,
            )
            step_hist_2, _ = hist_from_tags(
                readout2_tags,
                detect_ns=detect_ns,
                num_bins=num_bins,
            )

            hist_readout_1[step_ind] += step_hist_1
            hist_readout_2[step_ind] += step_hist_2

            int_counts_1[run_ind, step_ind] = integrate_histogram(
                step_hist_1,
                bin_centers_ns,
                integrate_start_ns=integrate_start_ns,
                integrate_end_ns=integrate_end_ns,
                subtract_tail_bg=subtract_tail_bg,
            )

            int_counts_2[run_ind, step_ind] = integrate_histogram(
                step_hist_2,
                bin_centers_ns,
                integrate_start_ns=integrate_start_ns,
                integrate_end_ns=integrate_end_ns,
                subtract_tail_bg=subtract_tail_bg,
            )

            print(
                f"    gates={num_processed_gates}, "
                f"readout1={int_counts_1[run_ind, step_ind]:.1f}, "
                f"readout2={int_counts_2[run_ind, step_ind]:.1f}"
            )

            if do_plot:
                mean_1 = np.nanmean(int_counts_1, axis=0)
                mean_2 = np.nanmean(int_counts_2, axis=0)

                ratio = np.divide(
                    mean_2,
                    mean_1,
                    out=np.full_like(mean_2, np.nan),
                    where=mean_1 > 0,
                )

                line_counts_1.set_data(effective_delay_ns_list, mean_1)
                line_counts_2.set_data(effective_delay_ns_list, mean_2)
                line_ratio.set_data(effective_delay_ns_list, ratio)

                ax_counts.relim()
                ax_counts.autoscale_view()

                ax_ratio.relim()
                ax_ratio.autoscale_view()

                plt.pause(0.01)

        print(f"Elapsed time: {time.time() - start_time:.2f} s")

    mean_counts_1 = np.nanmean(int_counts_1, axis=0)
    mean_counts_2 = np.nanmean(int_counts_2, axis=0)

    ste_counts_1 = np.nanstd(int_counts_1, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(np.isfinite(int_counts_1), axis=0), 1)
    )
    ste_counts_2 = np.nanstd(int_counts_2, axis=0, ddof=1) / np.sqrt(
        np.maximum(np.sum(np.isfinite(int_counts_2), axis=0), 1)
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

    # Use the longest effective delay for representative lifetime traces.
    best_ind = int(np.nanargmax(effective_delay_ns_list))

    lifetime_fit_1 = None
    lifetime_fit_2 = None

    if bin_centers_ns is not None:
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

    recovery_fit = fit_recovery(
        effective_delay_ns_list,
        ratio_mean,
        ratio_ste=ratio_ste,
    )

    if do_plot and bin_centers_ns is not None:
        ax_lifetime.clear()

        ax_lifetime.plot(
            bin_centers_ns,
            hist_readout_1[best_ind],
            "o-",
            ms=3,
            label=(
                f"readout 1, effective delay = "
                f"{effective_delay_ns_list[best_ind]} ns"
            ),
        )

        ax_lifetime.plot(
            bin_centers_ns,
            hist_readout_2[best_ind],
            "o-",
            ms=3,
            label=(
                f"readout 2, effective delay = "
                f"{effective_delay_ns_list[best_ind]} ns"
            ),
        )

        xfine_life = np.linspace(
            np.nanmin(bin_centers_ns),
            np.nanmax(bin_centers_ns),
            500,
        )

        if lifetime_fit_1 is not None:
            tau1 = lifetime_fit_1["popt"][1]
            ax_lifetime.plot(
                xfine_life,
                exp_decay_with_bg(xfine_life, *lifetime_fit_1["popt"]),
                "-",
                label=f"fit 1 tau = {tau1:.2f} ns",
            )

        if lifetime_fit_2 is not None:
            tau2 = lifetime_fit_2["popt"][1]
            ax_lifetime.plot(
                xfine_life,
                exp_decay_with_bg(xfine_life, *lifetime_fit_2["popt"]),
                "-",
                label=f"fit 2 tau = {tau2:.2f} ns",
            )

        ax_lifetime.set_xlabel("Photon arrival time after APD gate opens (ns)")
        ax_lifetime.set_ylabel("Counts")
        ax_lifetime.legend()

        if recovery_fit is not None:
            xfine_rec = np.linspace(
                np.nanmin(effective_delay_ns_list),
                np.nanmax(effective_delay_ns_list),
                500,
            )
            yfine_rec = recovery_model(xfine_rec, *recovery_fit["popt"])

            tau_rec = recovery_fit["popt"][2]

            ax_ratio.plot(
                xfine_rec,
                yfine_rec,
                "-",
                label=f"recovery fit tau = {tau_rec:.2f} ns",
            )
            ax_ratio.legend()

        plt.pause(0.01)

    proc_data = {
        "recovery_delay_ns_list": recovery_delay_ns_list.tolist(),
        "effective_delay_ns_list": effective_delay_ns_list.tolist(),
        "bin_centers_ns": bin_centers_ns.tolist()
        if bin_centers_ns is not None
        else None,
        "hist_readout_1": hist_readout_1.tolist(),
        "hist_readout_2": hist_readout_2.tolist(),
        "mean_counts_1": mean_counts_1.tolist(),
        "mean_counts_2": mean_counts_2.tolist(),
        "ste_counts_1": ste_counts_1.tolist(),
        "ste_counts_2": ste_counts_2.tolist(),
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
            "tau_recovery_ns": float(recovery_fit["popt"][2]),
            "R_inf_err": float(recovery_fit["perr"][0]),
            "A_err": float(recovery_fit["perr"][1]),
            "tau_recovery_ns_err": float(recovery_fit["perr"][2]),
        },
    }

    raw_data = {
        "timestamp": timestamp,
        "sample_name": getattr(sample_sig, "name", "sample"),
        "num_reps": int(num_reps),
        "num_runs": int(num_runs),
        "exc_ns": int(exc_ns),
        "readout_delay_ns": int(readout_delay_ns),
        "detect_ns": int(detect_ns),
        "num_bins": int(num_bins),
        "laser_vkey": str(laser_vkey),
        "laser_power": laser_power,
        "min_recovery_delay_ns": int(min_recovery_delay_ns),
        "max_recovery_delay_ns": int(max_recovery_delay_ns),
        "num_steps": int(num_steps),
        "recovery_delay_ns_list": recovery_delay_ns_list.tolist(),
        "effective_delay_ns_list": effective_delay_ns_list.tolist(),
        "int_counts_1": int_counts_1.tolist(),
        "int_counts_2": int_counts_2.tolist(),
        "integrate_start_ns": integrate_start_ns,
        "integrate_end_ns": integrate_end_ns,
        "subtract_tail_bg": bool(subtract_tail_bg),
        "randomize_delay_order": bool(randomize_delay_order),
    }

    if do_save:
        file_path = dm.get_file_path(
            __file__,
            timestamp,
            getattr(sample_sig, "name", "sample"),
        )
        dm.save_raw_data(raw_data, file_path)
        dm.save_raw_data(proc_data, file_path + "_proc")

        if fig is not None:
            dm.save_figure(fig, file_path)

        print(f"Saved data to {file_path}")

    if proc_data["recovery_fit"] is not None:
        print(
            "Recovery tau = "
            f"{proc_data['recovery_fit']['tau_recovery_ns']:.3f} ± "
            f"{proc_data['recovery_fit']['tau_recovery_ns_err']:.3f} ns"
        )

    if proc_data["lifetime_fit_1"] is not None:
        print(
            "Lifetime readout 1 tau = "
            f"{proc_data['lifetime_fit_1']['tau_ns']:.3f} ± "
            f"{proc_data['lifetime_fit_1']['tau_ns_err']:.3f} ns"
        )

    if proc_data["lifetime_fit_2"] is not None:
        print(
            "Lifetime readout 2 tau = "
            f"{proc_data['lifetime_fit_2']['tau_ns']:.3f} ± "
            f"{proc_data['lifetime_fit_2']['tau_ns_err']:.3f} ns"
        )

    tb.reset_cfm()

    return raw_data, proc_data


if __name__ == "__main__":

    class Dummy:
        name = "double_lifetime_recovery_test"

    sample_sig = Dummy()

    raw_data, proc_data = main(
        sample_sig=sample_sig,
        num_reps=100000,
        num_runs=5,

        # Sweep extra dark time after readout 1
        min_recovery_delay_ns=0,
        max_recovery_delay_ns=50000,
        num_steps=31,

        # Lifetime timing
        exc_ns=50,
        readout_delay_ns=0,
        detect_ns=500,
        num_bins=100,

        laser_vkey="SPIN_READOUT",
        laser_power=None,

        # Integrate early part or full detection window.
        # For full window, leave integrate_end_ns=None.
        integrate_start_ns=0,
        integrate_end_ns=300,

        # Try False first. Then compare with True.
        subtract_tail_bg=False,

        # Lifetime fit region
        fit_lifetime_start_ns=0,
        fit_lifetime_end_ns=300,

        randomize_delay_order=True,
        do_plot=True,
        do_save=False,
    )