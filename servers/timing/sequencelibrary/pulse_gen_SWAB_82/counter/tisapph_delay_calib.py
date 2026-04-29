# -*- coding: utf-8 -*-
"""
Ti:sapph hardware delay calibration sequence.

Goal: find the true total delay (cable + AOM driver + acoustic + optical) between
issuing a Ti:sapph TTL command on the Pulse Streamer and the Ti:sapph light
arriving at the sample, by sweeping the commanded Ti:sapph pulse position
relative to the green readout pulse and looking for a count dip when the two
overlap at the sample.

Important design choice:
    The Ti:sapph train is placed in the sequence WITHOUT any delay compensation
    (no use of tisapph_aom_delay from config). The green readout train IS delay-
    compensated as usual (via tb.process_laser_seq). The dip center in
    delta_ns then directly gives the true Ti:sapph delay relative to the green
    readout chain. Since green readout_delay is typically 0, the dip center IS
    the true tisapph_aom_delay.

Two APD-gated experiments per repetition:
    gate 0 = reference  (Ti:sapph OFF)
    gate 1 = signal     (Ti:sapph pulse at swept offset)

Args:
    [pol_ns, tisapph_pulse_ns, readout_ns, delta_ns, spin_pol_vkey, readout_vkey]

    delta_ns: signed offset of the Ti:sapph pulse leading edge relative to the
              readout pulse leading edge, in commanded sequence time.
              delta_ns < 0  -> Ti:sapph fires before readout
              delta_ns = 0  -> Ti:sapph fires coincident with readout (commanded)
              delta_ns > 0  -> Ti:sapph fires after readout
"""

from pulsestreamer import Sequence, OutputState
import numpy as np

from utils import tool_belt as tb
from utils.constants import Digital, VirtualLaserKey

LOW = Digital.LOW
HIGH = Digital.HIGH


def _as_int64(name, v):
    try:
        iv = int(v)
    except Exception:
        raise TypeError(f"{name} must be int-like, got {type(v).__name__}: {v!r}")
    return np.int64(iv)


def _vkey_from_arg(x):
    if isinstance(x, VirtualLaserKey):
        return x
    if isinstance(x, str):
        name = x.split(".")[-1]
        try:
            return VirtualLaserKey[name]
        except Exception:
            return VirtualLaserKey(x)
    raise TypeError(f"Bad virtual laser key: {x!r}")


def get_seq(pulse_streamer, config, args, num_reps=1):
    (pol_ns, tisapph_pulse_ns, readout_ns, delta_ns,
     spin_pol_vkey_arg, readout_vkey_arg) = args

    pol_ns = _as_int64("pol_ns", pol_ns)
    tisapph_pulse_ns = _as_int64("tisapph_pulse_ns", tisapph_pulse_ns)
    if tisapph_pulse_ns < 0:
        raise ValueError("tisapph_pulse_ns must be >= 0")
    readout_ns = _as_int64("readout_ns", readout_ns)
    if readout_ns <= 0:
        raise ValueError("readout_ns must be > 0")
    delta_ns = _as_int64("delta_ns", delta_ns)  # signed

    spin_pol_vkey = _vkey_from_arg(spin_pol_vkey_arg)
    readout_vkey = _vkey_from_arg(readout_vkey_arg)

    pulser_wiring = config["Wiring"]["PulseGen"]
    do_sample_clock = pulser_wiring["do_sample_clock"]
    do_apd_gate = pulser_wiring["do_apd_gate"]
    do_tisapph_aom = pulser_wiring["do_laser_TISAPPH_dm"]

    # Laser delays / physical names
    spin_pol_laser_name = tb.get_physical_laser_name(spin_pol_vkey)
    readout_laser_name = tb.get_physical_laser_name(readout_vkey)
    spin_pol_delay = _as_int64(
        "spin_pol_delay",
        config["Optics"]["PhysicalLasers"][spin_pol_laser_name]["delay"],
    )
    readout_delay = _as_int64(
        "readout_delay",
        config["Optics"]["PhysicalLasers"][readout_laser_name]["delay"],
    )

    meas_buffer = _as_int64("meas_buffer", config["CommonDurations"]["cw_meas_buffer"])
    transient = np.int64(1000)

    # We build everything in "logical" time (when light arrives at sample for green;
    # when TTL is asserted for Ti:sapph, since we are NOT compensating Ti:sapph).
    # process_laser_seq will later shift the green train by readout_delay.

    # The block layout in logical time:
    #
    #   [ pol_ns ][ transient ][ tisapph_lead ][ readout_ns ][ tisapph_trail ][ meas_buffer ]
    #
    # where the readout pulse starts at t = pol + transient + tisapph_lead within
    # the block. The Ti:sapph pulse, in commanded sequence time, starts at
    # readout_start + delta_ns and ends at readout_start + delta_ns + tisapph_pulse_ns.
    #
    # We need the Ti:sapph pulse to fit entirely within the block, so we pad
    # before/after with enough room to accommodate the swept range.
    #
    # tisapph_lead and tisapph_trail bound the position of the Ti:sapph pulse so
    # that no matter where delta_ns puts it (within +/- max_excursion), it fits.

    # Choose padding: enough to cover the full sweep range comfortably
    pad = np.int64(max(abs(int(delta_ns)) + int(tisapph_pulse_ns) + 200, 2000))

    # Time within block (logical, relative to block start):
    t_pol_start    = np.int64(0)
    t_pol_end      = t_pol_start + pol_ns
    t_pad1_end     = t_pol_end + transient + pad
    t_readout_start = t_pad1_end
    t_readout_end  = t_readout_start + readout_ns
    t_pad2_end     = t_readout_end + pad
    t_block_end    = t_pad2_end + meas_buffer

    block_ns = t_block_end  # total block length

    # Front buffer: must be large enough that no train ever goes negative,
    # given that process_laser_seq for green will pull the green train EARLIER
    # in TTL time by readout_delay, but we don't apply any shift to Ti:sapph or APD.
    front_buffer = np.int64(max(spin_pol_delay, readout_delay, 0))

    # Period: front_buffer + 2 blocks (ref + sig)
    period = np.int64(front_buffer + 2 * block_ns)

    seq = Sequence()

    # ------------------------------------------------------------------
    # Sample clock - one tick at the very end
    # ------------------------------------------------------------------
    if period >= 300:
        clk_train = [(int(period - 200), LOW), (100, HIGH), (100, LOW)]
    else:
        clk_train = [(int(period), LOW)]
    seq.setDigital(do_sample_clock, clk_train)

    # ------------------------------------------------------------------
    # APD gate - HIGH only during the readout window of each block
    # ------------------------------------------------------------------
    apd_train = [(int(front_buffer), LOW)]
    for _ in range(2):
        apd_train.extend([
            (int(t_readout_start), LOW),                  # before readout
            (int(readout_ns), HIGH),                      # gate open
            (int(block_ns - t_readout_end), LOW),         # after readout
        ])
    seq.setDigital(do_apd_gate, apd_train)

    # ------------------------------------------------------------------
    # Ti:sapph AOM TTL - placed WITHOUT delay compensation
    # Block 0: OFF (reference)
    # Block 1: pulse at readout_start + delta_ns (in commanded TTL time)
    # ------------------------------------------------------------------
    # Tisapph pulse start, in absolute commanded time, is:
    #   front_buffer + block_ns + t_readout_start + delta_ns
    # We need delta_ns to be allowed negative; require it to be > -t_readout_start
    # and the pulse end < block_ns. Sanity check:
    sig_block_pulse_start = t_readout_start + int(delta_ns)
    sig_block_pulse_end = sig_block_pulse_start + int(tisapph_pulse_ns)
    if sig_block_pulse_start < 0:
        raise ValueError(
            f"delta_ns={int(delta_ns)} too negative; "
            f"would place Ti:sapph pulse start at {sig_block_pulse_start} "
            f"within signal block (must be >= 0). Increase pad or reduce sweep range."
        )
    if sig_block_pulse_end > block_ns:
        raise ValueError(
            f"delta_ns={int(delta_ns)} too positive; Ti:sapph pulse "
            f"would extend past block end ({sig_block_pulse_end} > {block_ns}). "
            f"Increase pad or reduce sweep range."
        )

    tisapph_train = [
        (int(front_buffer), LOW),
        # ---- block 0: reference, Ti:sapph OFF ----
        (int(block_ns), LOW),
        # ---- block 1: signal, Ti:sapph pulse at swept offset ----
        (int(sig_block_pulse_start), LOW),
        (int(tisapph_pulse_ns), HIGH if tisapph_pulse_ns > 0 else LOW),
        (int(block_ns - sig_block_pulse_end), LOW),
    ]
    # Drop zero-duration entries (pulse=0 case)
    tisapph_train = [(d, lvl) for (d, lvl) in tisapph_train if d > 0]
    seq.setDigital(do_tisapph_aom, tisapph_train)

    # ------------------------------------------------------------------
    # Green laser train (polarization + readout, same physical channel)
    # Routed through process_laser_seq so AOM delay compensation is applied.
    # ------------------------------------------------------------------
    if spin_pol_laser_name != readout_laser_name:
        raise NotImplementedError(
            "This calibration assumes spin_pol and readout share the same "
            "physical laser channel (green). Got: "
            f"{spin_pol_laser_name} vs {readout_laser_name}"
        )

    shared_delay = max(int(spin_pol_delay), int(readout_delay))
    green_train = [(int(front_buffer - shared_delay), LOW)]
    for _ in range(2):
        green_train.extend([
            (int(pol_ns), HIGH),                           # polarization
            (int(t_readout_start - t_pol_end), LOW),       # transient + pad1
            (int(readout_ns), HIGH),                       # readout
            (int(block_ns - t_readout_end), LOW),          # pad2 + meas_buffer
        ])
    green_train.append((int(shared_delay), LOW))
    tb.process_laser_seq(seq, readout_vkey, green_train)

    final = OutputState([], 0.0, 0.0)
    return seq, final, [int(period)]


if __name__ == "__main__":
    from utils import common
    cfg = common.get_config_dict()
    # Example: pol=2us, tisapph_pulse=610ns, readout=610ns, delta=0
    args = [2000, 610, 610, 0, "SPIN_POL", "SPIN_READOUT"]
    seq, final, ret = get_seq(None, cfg, args)
    print("Period (ns):", ret[0])
    seq.plot()
