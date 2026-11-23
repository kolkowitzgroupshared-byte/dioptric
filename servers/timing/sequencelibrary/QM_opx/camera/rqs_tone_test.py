# -*- coding: utf-8 -*-
"""
Test of the RQS test-field tones: plays up to five tones at once, continuously, on
analog output 1 (the ao_rqs_tone_<k> elements), for checking the kHz chain with a
scope or spectrum analyzer.

seq_args: [tones], with tones a list of [frequency_hz, amp_v, phase_deg], the same
format as rqs.main. amp_v is the amplitude into 50 ohm at the OPX output; together
the tones must stay within 0.5 V. num_reps = -1 plays until the OPX is halted.

Control panel function to run it (copy it from this file, not from a chat):

def do_test_rqs_tones(tones=((60e3, 0.1, 0.0), (100e3, 0.1, 0.0), (140e3, 0.1, 0.0))):
    cxn = common.labrad_connect()
    pulse_gen = cxn.QM_opx
    seq_args_string = tb.encode_seq_args([[list(tone) for tone in tones]])
    pulse_gen.stream_immediate("rqs_tone_test.py", seq_args_string, -1)
    input("Press enter to stop...")
    pulse_gen.halt()

Created on November 6th, 2026

@author: schand
"""

from qm import qua

from servers.timing.sequencelibrary.QM_opx.camera import rqs

PLAY_CC = 25000  # each tone is played in back-to-back 100 us pieces; the phase runs on without breaks


def get_seq(tones, num_reps=-1):
    # Same checks and amplitude scaling as the RQS sequence: elements exist, sum within 0.5 V
    settings = rqs.tone_settings(tones, 0, phase_locked=False)
    if not settings:
        raise ValueError("No tones to play")

    with qua.program() as seq:
        for element, freq, _, turns in settings:
            qua.update_frequency(element, freq)
            qua.reset_frame(element)
            qua.frame_rotation_2pi(turns, element)

        def play_tones():
            for element, _, scale, _ in settings:
                qua.play("cw" * qua.amp(scale), element, duration=PLAY_CC)

        if num_reps == -1:
            with qua.infinite_loop_():
                play_tones()
        else:
            rep = qua.declare(int)
            with qua.for_(rep, 0, rep < num_reps, rep + 1):
                play_tones()

    seq_ret_vals = []
    return seq, seq_ret_vals
