# -*- coding: utf-8 -*-
"""
Rabi sequence (Pulse Streamer): two APD gates per period (signal, then reference).
Constant period per step by padding with (pad_budget_ns - tau_ns).

Author: Saroj Chand
"""

# from typing import Callable, List, Tuple

# from pulsestreamer import OutputState, Sequence

# from servers.timing.sequencelibrary.pulse_gen_SWAB_82 import seq_utils
# from servers.timing.sequencelibrary.pulse_gen_SWAB_82.counter import base_sequence
# from utils import tool_belt as tb
# from utils.constants import VirtualLaserKey

# MWMacro = Callable[[seq_utils.PSBuilder, int, int], int]


# def _pick_align_index(uwave_inds: List[int], readout_laser: str) -> int:
#     """Use the chain with the largest MW delay for base_args alignment."""
#     best = uwave_inds[0]
#     _, best_udel, _, _, _ = seq_utils._delays(readout_laser, best)
#     for ind in uwave_inds[1:]:
#         _, udel, _, _, _ = seq_utils._delays(readout_laser, ind)
#         if udel > best_udel:
#             best, best_udel = ind, udel
#     return best


# def uwave_rabi(uwave_inds: List[int], readout_laser: str) -> MWMacro:
#     """Gate all MW chains in uwave_inds simultaneously for τ."""
#     # collect per-chain MW delays and get a shared t0
#     udel = {}
#     ldel_list = []
#     short = 10
#     for ind in uwave_inds:
#         ldel_i, udel_i, _t0, _uwbuf, short = seq_utils._delays(readout_laser, ind)
#         ldel_list.append(ldel_i)
#         udel[ind] = udel_i
#     # laser delay for readout path (same laser for both experiments)
#     # ldel = ldel_list[0]
#     ldel = 0
#     t0 = max([ldel] + list(udel.values())) + short

#     def _fn(b: seq_utils.PSBuilder, t: int, tau_ns: int) -> int:
#         for ind in uwave_inds:
#             seq_utils.macro_mw_pulse(
#                 b,
#                 uwave_ind=ind,
#                 start_ns=t,
#                 dur_ns=int(tau_ns),
#                 uwave_delay_ns=udel[ind],
#                 t0_ns=t0,
#             )
#         return t + int(tau_ns)

#     return _fn


# def uwave_ref() -> MWMacro:
#     def _fn(b: seq_utils.PSBuilder, t: int, _tau_ns: int) -> int:
#         return t

#     return _fn


# def get_seq(_server, _config, args) -> Tuple[Sequence, OutputState, List[int]]:
#     """
#     args:
#       0: base_args = [pol_ns, readout_ns, uwave_ind_align, readout_laser, readout_power, pad_budget_ns]
#       1: step_tau_ns (int)
#       2: num_reps_ignored (int)
#       3: uwave_inds (optional list[int])  # if omitted, falls back to [uwave_ind_align]
#     """
#     base_args, step_tau_ns, _num_reps = args[:3]
#     uwave_inds = args[3] if len(args) >= 4 else None

#     pol_ns, readout_ns, uwave_ind_align, ro_laser, ro_power, pad_budget_ns = base_args

#     # If caller didn't pass a list, use the align chain as the single source.
#     # if uwave_inds is None:
#     #     uwave_inds = uwave_ind_align
#     # else:
#     #     uwave_inds = [int(i) for i in uwave_inds]

#     # Ensure base_args uses the chain with the largest delay for alignment
#     align_ind = _pick_align_index(uwave_inds, ro_laser)
#     base_args = [
#         int(pol_ns),
#         int(readout_ns),
#         int(align_ind),
#         str(ro_laser),
#         (None if ro_power is None else float(ro_power)),
#         int(pad_budget_ns),
#     ]

#     # Two experiments total: (signal with all chains) + (reference)
#     uwave_macros = [uwave_rabi(uwave_inds, ro_laser), uwave_ref()]

#     return base_sequence.macro(
#         base_args=base_args,
#         uwave_macros=uwave_macros,
#         step_val_ns=int(step_tau_ns),
#         num_reps_ignored=int(_num_reps),
#         include_reference=False,
#     )

# Optional local preview
# if __name__ == "__main__":
#     tau_ns = 200
#     pol_ns = 1000
#     readout_ns = 300
#     pad_budget_ns = 220  # constant-evolution padding budget per period
#     ro_laser = tb.get_physical_laser_name(VirtualLaserKey.WIDEFIELD_CHARGE_READOUT)
#     ro_power = None  # None if digital; float volts if analog
#     # use both chains:
#     uwave_inds = [0, 1]
#     align_ind = _pick_align_index(uwave_inds, ro_laser)

#     base_args = [pol_ns, readout_ns, align_ind, ro_laser, ro_power, pad_budget_ns]
#     args = [base_args, tau_ns, 1, uwave_inds]

#     seq, _final, (period_ns,) = get_seq(None, None, args)
#     print(
#         f"[RABI PREVIEW] period = {period_ns} ns  (tau = {tau_ns} ns, MW={uwave_inds})"
#     )
#     # built-in plotter
#     import matplotlib.pyplot as plt

#     seq.plot()
#     plt.show()


# -*- coding: utf-8 -*-
"""
Confocal Rabi pulse-streamer sequence (SWAB_82 counter path)

Contract:
  args = [base_args, tau] or [base_args, tau, num_reps_ignored]
  base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau_ns]

Sequence produces EXACTLY 2 APD gates per repetition:
  gate0 = signal readout
  gate1 = reference readout

Local preview:
  python rabi_seq.py  (will plot)
"""

import numpy as np
from pulsestreamer import OutputState, Sequence

from utils import common
from utils import tool_belt as tb
from utils.constants import Digital, ModMode, VirtualLaserKey

LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64(name, v):
    try:
        iv = int(v)
    except Exception:
        raise TypeError(f"{name} must be int-like, got {type(v).__name__}: {v!r}")
    if iv < 0:
        raise ValueError(f"{name} must be >= 0, got {iv}")
    return np.int64(iv)


def _vkey_from_arg(x):
    """
    Accepts:
      - VirtualLaserKey member
      - "SPIN_READOUT"
      - "VirtualLaserKey.SPIN_READOUT"
      - value string (if VirtualLaserKey is str-enum)
    Returns VirtualLaserKey
    """
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        name = x.split(".")[-1]
        # try by Enum name
        try:
            return VirtualLaserKey[name]
        except Exception:
            # try by Enum value
            return VirtualLaserKey(x)
    raise TypeError(f"Bad virtual laser key: {x!r}")


def _set_laser_train(seq: Sequence, cfg: dict, readout_vkey: VirtualLaserKey, train, readout_power=None):
    """Handle DIGITAL vs ANALOG modulation for the selected virtual laser."""
    wiring = cfg["Wiring"]["PulseGen"]

    vld = cfg["Optics"]["VirtualLasers"][readout_vkey]
    laser_name = vld["physical_name"]
    pld = cfg["Optics"]["PhysicalLasers"][laser_name]
    mod_mode = pld["mod_mode"]

    if mod_mode is ModMode.DIGITAL:
        do_chan = wiring[f"do_{laser_name}_dm"]
        seq.setDigital(do_chan, train)
        return

    if mod_mode is ModMode.ANALOG:
        ao_chan = wiring[f"ao_{laser_name}_am"]
        # Choose power: explicit arg wins; else use virtual laser config.
        if readout_power is None:
            if "laser_power" not in vld:
                raise ValueError(
                    f"{readout_vkey} is ANALOG but no readout_power provided and no "
                    f"'laser_power' in config Optics->VirtualLasers->{readout_vkey}."
                )
            readout_power = vld["laser_power"]

        power = float(readout_power)
        power_dict = {LOW: 0.0, HIGH: power}
        processed = [(int(d), power_dict[val]) for (d, val) in train]
        seq.setAnalog(ao_chan, processed)
        return

    raise ValueError(f"Unknown mod_mode for {laser_name}: {mod_mode!r}")


def get_seq(pulse_streamer, config, args):
    # -------- parse args --------
    if len(args) == 3 and isinstance(args[0], (list, tuple)):
        base_args, tau, _num_reps_ignored = args
    elif len(args) == 2 and isinstance(args[0], (list, tuple)):
        base_args, tau = args
    else:
        raise ValueError(
            "Expected args as [base_args, tau] or [base_args, tau, num_reps]. "
            f"Got: {args!r}"
        )

    # base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau_ns]
    # if len(base_args) != 6:
    #     raise ValueError(f"base_args must have length 6; got {len(base_args)}: {base_args!r}")

    pol_ns, readout_ns, uwave_ind_list, readout_vkey_arg, readout_power, max_tau_ns = base_args

    pol_ns = _as_int64("pol_ns", pol_ns)
    readout_ns = _as_int64("readout_ns", readout_ns)
    tau = _as_int64("tau", tau)
    max_tau_ns = _as_int64("max_tau_ns", max_tau_ns)

    # inside rabi_seq.py
    if tau > max_tau_ns:
        max_tau_ns = tau  


    pad_ns = _as_int64("pad_ns", (max_tau_ns - tau))

    # normalize uwave_ind_list
    if isinstance(uwave_ind_list, (int, np.integer)):
        uwave_ind_list = [int(uwave_ind_list)]
    else:
        uwave_ind_list = [int(x) for x in uwave_ind_list]

    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    # -------- wiring & delays --------
    wiring = config["Wiring"]["PulseGen"]
    do_apd_gate = wiring["do_apd_gate"]
    do_sample_clock = wiring["do_sample_clock"]

    # delays
    laser_name = config["Optics"]["VirtualLasers"][readout_vkey]["physical_name"]
    laser_delay = _as_int64("laser_delay", config["Optics"]["PhysicalLasers"][laser_name]["delay"])

    uwave_delays = []
    do_sig_gen_dm_list = []
    for uwave_ind in uwave_ind_list:
        vsg = tb.get_virtual_sig_gen_dict(uwave_ind)
        sig_gen_name = vsg["physical_name"]
        do_sig_gen_dm_list.append(wiring[f"do_{sig_gen_name}_dm"])
        uwave_delays.append(config["Microwaves"]["PhysicalSigGens"][sig_gen_name]["delay"])

    uwave_delay = _as_int64("uwave_delay", max(uwave_delays) if uwave_delays else 0)
    common_delay = np.int64(max(laser_delay, uwave_delay))

    uwave_buffer = _as_int64("uwave_buffer", config["CommonDurations"]["uwave_buffer"])

    # -------- timeline per repetition --------
    # 0) common_delay
    # 1) polarization (laser ON) pol_ns
    # 2) buffer
    # 3) signal evolution max_tau_ns: mw ON for tau then OFF for pad
    # 4) buffer
    # 5) readout (laser ON, APD ON) readout_ns
    # 6) buffer
    # 7) reference evolution max_tau_ns: mw OFF
    # 8) buffer
    # 9) readout2 (laser ON, APD ON) readout_ns
    # 10) buffer (small dead time)

    # APD gate: only during the two readouts
    apd_train = [
        (common_delay, LOW),
        (pol_ns, LOW),
        (uwave_buffer, LOW),
        (max_tau_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (uwave_buffer, LOW),
        (max_tau_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (uwave_buffer, LOW),
    ]

    # Laser: shift earlier by laser_delay (start with LOW for common_delay-laser_delay)
    laser_train = [
        (common_delay - laser_delay, LOW),
        (pol_ns, HIGH),
        (uwave_buffer, LOW),
        (max_tau_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (uwave_buffer, LOW),
        (max_tau_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, HIGH),
        (uwave_buffer + laser_delay, LOW),
    ]

    # MW: shift earlier by uwave_delay
    mw_train = [
        (common_delay - uwave_delay, LOW),
        (pol_ns, LOW),
        (uwave_buffer, LOW),
        (tau, HIGH),
        (pad_ns, LOW),
        (uwave_buffer, LOW),
        (readout_ns, LOW),
        (uwave_buffer, LOW),
        (max_tau_ns, LOW),   # reference: OFF
        (uwave_buffer, LOW),
        (readout_ns, LOW),
        (uwave_buffer + uwave_delay, LOW),
    ]

    # -------- assemble --------
    seq = Sequence()
    seq.setDigital(do_apd_gate, apd_train)
    _set_laser_train(seq, config, readout_vkey, laser_train, readout_power=readout_power)

    for do_sig_gen_dm in do_sig_gen_dm_list:
        seq.setDigital(do_sig_gen_dm, mw_train)

    period_ns = np.int64(sum(int(d) for d, _ in apd_train))

    # sample clock: 100 ns pulse near end (if long enough)
    if period_ns >= 300:
        clk_train = [(int(period_ns - 200), LOW), (100, HIGH), (100, LOW)]
    else:
        clk_train = [(int(period_ns), LOW)]
    seq.setDigital(do_sample_clock, clk_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period_ns)]


if __name__ == "__main__":
    cfg = common.get_config_dict()

    pol_ns = 10_000
    readout_ns = 300
    uwave_ind_list = [0]
    readout_vkey = "SPIN_READOUT"
    readout_power = None
    max_tau_ns = 400
    tau = 235

    base_args = [pol_ns, readout_ns, uwave_ind_list, readout_vkey, readout_power, max_tau_ns]
    seq, final, ret = get_seq(None, cfg, [base_args, tau])

    print("Period (ns):", int(ret[0]))
    seq.plot()
