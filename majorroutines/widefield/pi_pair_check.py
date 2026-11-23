# -*- coding: utf-8 -*-
"""
Does a second pi pulse act on the spin? A population test, insensitive to dephasing.

Steps: no pulse, one pi pulse, and two pi pulses with each gap in gaps_ns between them. One pi
flips the spin from m_s = 0 to m_s = -1; a second pi flips it back, whatever the gap (T1 is
far longer). The NV-averaged signal of each step is compared with the two references:

    flipped back = (single - pair) / (single - none)

About 1 (or a bit less, since the pi pulse is not perfect) means the second pulse acts; about
0 means it does not, and the pair reads like a single pulse.

Control panel:

def do_pi_pair_check(nv_list):
    pi_pair_check.main(nv_list, num_reps=10, num_runs=20, gaps_ns=(100, 1000, 10000, 30000), iq_phase=0)
    pi_pair_check.main(nv_list, num_reps=10, num_runs=20, gaps_ns=(100, 1000, 10000, 30000), iq_phase=None)

iq_phase: IQ phase (deg) of the pi pulses, which turns IQ mode on; None runs in CW mode.

Created on November 18th, 2026
"""

import numpy as np

EXCLUDE_NVS = (0, 6)  # another orientation
THRESHOLD = True


def analyze(raw_data, plot=True, exclude_nvs=EXCLUDE_NVS):
    from utils import widefield

    nv_list = raw_data["nv_list"]
    gaps = list(raw_data["gaps_ns"])
    keep = [i for i in range(len(nv_list)) if i not in set(exclude_nvs)]
    counts = np.asarray(raw_data["counts"], dtype=np.float64)
    counts = np.where(np.isfinite(counts), counts, np.nan)
    if np.isnan(counts).any():  # rare float16 overflows: replace by the median of the same NV and step
        fill = np.nanmedian(counts, axis=(2, 4), keepdims=True)
        counts = np.where(np.isnan(counts), np.broadcast_to(fill, counts.shape), counts)
    processed = widefield.process_counts([nv_list[i] for i in keep], counts[0][keep], counts[1][keep],
                                         threshold=THRESHOLD)
    avg, ste = np.asarray(processed[0], float), np.asarray(processed[1], float)
    w = np.where(np.isfinite(ste) & (ste > 0) & np.isfinite(avg), 1 / np.where(ste > 0, ste, 1) ** 2, 0.0)
    mean = (np.where(w > 0, avg, 0) * w).sum(axis=0) / w.sum(axis=0)
    err = 1 / np.sqrt(w.sum(axis=0))

    none, single = mean[0], mean[1]
    span = single - none
    span_err = np.hypot(err[0], err[1])
    mode = "CW" if raw_data.get("iq_phase") is None else f"IQ, phase {raw_data['iq_phase']:g} deg"
    print(f"\nTwo-pulse check ({mode}), {len(keep)} NVs (excluded {list(exclude_nvs)}), {counts.shape[2]} runs")
    print(f"  no pulse {none:.3f} +/- {err[0]:.3f}; one pi {single:.3f} +/- {err[1]:.3f} "
          f"(difference {span:+.3f} +/- {span_err:.3f})")
    back = []
    for k, gap in enumerate(gaps):
        pair, pair_err = mean[2 + k], err[2 + k]
        frac = (single - pair) / span if span != 0 else np.nan
        frac_err = np.hypot(pair_err, err[1]) / abs(span) if span != 0 else np.nan
        back.append((frac, frac_err))
        print(f"  two pi, gap {gap / 1e3:7.2f} us: {pair:.3f} +/- {pair_err:.3f} -> flipped back "
              f"{100 * frac:.0f} +/- {100 * frac_err:.0f} %")
    if abs(span) < 3 * span_err:
        print("  -> One pi pulse barely changes the signal: the pulses themselves do not act (check the microwave).")
    else:
        fr = np.array([b[0] for b in back])
        if np.all(fr > 0.5):
            print("  -> The second pulse acts at every gap: the multi-pulse failure is not a missing pulse.")
        elif np.all(fr < 0.3):
            print("  -> The second pulse does not act: pairs read like a single pi pulse.")
        else:
            print("  -> The second pulse acts at some gaps and not at others: see the plot.")

    results = {"none": float(none), "single": float(single), "pairs": mean[2:].tolist(),
               "pairs_err": err[2:].tolist(), "flipped_back": [float(b[0]) for b in back],
               "flipped_back_err": [float(b[1]) for b in back]}
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(2 + len(gaps))
        labels = ["no pulse", "one pi"] + [f"two pi\n{g / 1e3:g} us" for g in gaps]
        ax.errorbar(x, mean, yerr=err, fmt="o", ms=6)
        ax.axhline(none, color="tab:green", ls=":", lw=1, label="no pulse (m_s = 0)")
        ax.axhline(single, color="tab:red", ls=":", lw=1, label="one pi (flipped)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Normalized signal, NV average")
        ax.set_title(f"Does the second pi pulse act? ({mode})", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout() if fig.get_layout_engine() is None else None
        results["figure"] = fig
    return results


def main(nv_list, num_reps=10, num_runs=20, gaps_ns=(100, 1000, 10000, 30000), iq_phase=0, uwave_ind_list=(1,)):
    from majorroutines.widefield import base_routine
    from utils import data_manager as dm
    from utils import kplotlib as kpl
    from utils import tool_belt as tb
    from utils import widefield

    pulse_gen = tb.get_server_pulse_gen()
    seq_file = "pi_pair_check.py"
    uwave_ind_list = [int(ind) for ind in uwave_ind_list]
    gaps = [float(g) for g in gaps_ns]
    num_steps = 2 + len(gaps)

    def run_fn(shuffled_step_inds):
        seq_args = [
            widefield.get_base_scc_seq_args(nv_list, uwave_ind_list),
            [int(ind) for ind in shuffled_step_inds],
            gaps,
            None if iq_phase is None else float(iq_phase),
        ]
        pulse_gen.stream_load(seq_file, tb.encode_seq_args(seq_args), num_reps)

    raw_data = base_routine.main(nv_list, num_steps, num_reps, num_runs, run_fn,
                                 uwave_ind_list=uwave_ind_list, load_iq=iq_phase is not None)

    timestamp = dm.get_time_stamp()
    raw_data |= {
        "timestamp": timestamp,
        "gaps_ns": gaps,
        "iq_phase": None if iq_phase is None else float(iq_phase),
        "step_order": "0 = no pulse, 1 = one pi, 2 + k = two pi with gaps_ns[k] between them",
    }
    fig = None
    try:
        results = analyze(raw_data)
        fig = results.pop("figure", None)
        raw_data |= results
    except Exception as exc:
        print(f"Analysis failed ({exc}); the raw data is still saved.")

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    file_path = dm.get_file_path(__file__, timestamp, repr_nv_sig.name)
    dm.save_raw_data(raw_data, file_path)
    if fig is not None:
        dm.save_figure(fig, file_path)
    tb.reset_cfm()
    kpl.show()
    return raw_data
