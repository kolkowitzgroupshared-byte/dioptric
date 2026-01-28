# -*- coding: utf-8 -*-
"""
Control panel for the PC Nuclear

Created on August 5th, 2025

@author: mccambria
"""

### Imports
import os
import random
import sys
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import websocket

from majorroutines.spectroscopy import (
    singlet_search,
    singlet_search_1043,
    singlet_search_with_etalon,
    singlet_search_with_spect,
    singlet_search_with_spect_binned,
)

# from majorroutines import targeting
from utils import common
from utils import kplotlib as kpl
from utils import tool_belt as tb


def do_singlet_search():
    # min_wavelength = 750
    # max_wavelength = 850
    min_wavelength = 734
    max_wavelength = 740
    # num_steps = 1000
    # num_runs = 1
    num_steps = 15
    num_runs = 2

    singlet_search.main(min_wavelength, max_wavelength, num_steps, num_runs)


def do_singlet_search_with_etalon():
    # min_wavelength = 800
    # max_wavelength = 810
    # etalon_range = 20
    # etalon_spacing = 2
    # num_steps = 2
    # num_runs = 1
    # RF_on = True

    min_wavelength = 716
    max_wavelength = 800
    etalon_range = 90
    etalon_spacing = 1
    num_steps = 210
    num_runs = 1
    RF_on = False

    # min_wavelength = 854
    # max_wavelength = 855
    # etalon_range = 90
    # etalon_spacing = 1
    # num_steps = 2
    # num_runs = 1
    # RF_on = False

    singlet_search_with_etalon.main(
        min_wavelength,
        max_wavelength,
        num_steps,
        num_runs,
        RF_on,
        etalon_range,
        etalon_spacing,
    )


def do_singlet_search_with_etalon_1():
    # min_wavelength = 800
    # max_wavelength = 810
    # etalon_range = 20
    # etalon_spacing = 2
    # num_steps = 2
    # num_runs = 1
    # RF_on = True

    min_wavelength = 800
    max_wavelength = 900
    etalon_range = 90
    etalon_spacing = 1
    num_steps = 250
    num_runs = 1
    RF_on = False

    # min_wavelength = 854
    # max_wavelength = 855
    # etalon_range = 90
    # etalon_spacing = 1
    # num_steps = 2
    # num_runs = 1
    # RF_on = False

    singlet_search_with_etalon.main(
        min_wavelength,
        max_wavelength,
        num_steps,
        num_runs,
        RF_on,
        etalon_range,
        etalon_spacing,
    )


def do_singlet_search_with_etalon_2():
    # min_wavelength = 800
    # max_wavelength = 810
    # etalon_range = 20
    # etalon_spacing = 2
    # num_steps = 2
    # num_runs = 1
    # RF_on = True

    min_wavelength = 800
    max_wavelength = 805
    etalon_range = 90
    etalon_spacing = 1
    num_steps = 5
    num_runs = 1
    RF_on = False

    # min_wavelength = 854
    # max_wavelength = 855
    # etalon_range = 90
    # etalon_spacing = 1
    # num_steps = 2
    # num_runs = 1
    # RF_on = False

    singlet_search_with_etalon.main(
        min_wavelength,
        max_wavelength,
        num_steps,
        num_runs,
        RF_on,
        etalon_range,
        etalon_spacing,
    )


def do_singlet_search_1043():
    # min_wavelength = 850
    # max_wavelength = 860
    # etalon_range = 2
    # etalon_spacing = 1
    # num_steps = 3
    # num_runs = 2
    # RF_on = True
    # start_col_index = 1000
    # end_col_index = None
    # exposure_time = 500.0
    # tisaph_power = 300

    # min_wavelength = 768
    # max_wavelength = 769
    # etalon_range = 5
    # etalon_spacing = 1
    # num_steps = 2
    # num_runs = 1
    # RF_on = True
    # start_col_index = 1000
    # end_col_index = None
    # exposure_time = 1000.0
    # tisaph_power = 300

    min_wavelength = 750
    max_wavelength = 860
    etalon_range = 90
    etalon_spacing = 1
    num_steps = 275
    num_runs = 1
    RF_on = True
    start_col_index = 1000
    end_col_index = None
    exposure_time = 1000.0
    tisaph_power = 298

    # min_wavelength = 800
    # max_wavelength = 850
    # etalon_range = 90
    # etalon_spacing = 1
    # num_steps = 125
    # num_runs = 1
    # RF_on = True
    # start_col_index = None
    # end_col_index = None
    # exposure_time = 1000.0
    # tisaph_power = 335

    # singlet_search_with_spect.main(
    #     min_wavelength,
    #     max_wavelength,
    #     num_steps,
    #     num_runs,
    #     RF_on,
    #     etalon_range,
    #     start_col_index,
    #     end_col_index,
    #     exposure_time,
    #     etalon_spacing,
    #     tisaph_power,
    # )

    singlet_search_with_spect_binned.main(
        min_wavelength,
        max_wavelength,
        num_steps,
        num_runs,
        RF_on,
        etalon_range,
        start_col_index,
        end_col_index,
        exposure_time,
        etalon_spacing,
        tisaph_power,
    )


def test_sig_gen():
    server_name = "sig_gen_STAN_sg394"
    sig_gen = common.get_server_by_name(server_name)
    # sig_gen.set_freq(2.75)
    # sig_gen.uwave_off()
    sig_gen.uwave_on()


def test_shutter():
    server_name = "shutter_STAN_sr474"
    shutter = common.get_server_by_name(server_name)
    # shutter.enable(1)
    # shutter.disable(1)
    start = time.time()
    # shutter.open(1)
    shutter.close(1)
    print(time.time() - start)
    # shutter.close(1)


def test_multimeter():
    server_name = "multimeter_KEIT_daq6510"
    meter = common.get_server_by_name(server_name)
    print(meter.read())


def test_multimeter_stats():
    server_name = "multimeter_MULT_mp730028"
    meter = common.get_server_by_name(server_name)
    print(meter.get_stats())


def test_multimeter_duration():
    server_name = "multimeter_MULT_mp730028"
    meter = common.get_server_by_name(server_name)
    print(meter.meas_for_duration(0.4))


def test_tisapph():
    server_name = "tisapph_M2_solstis"
    tisapph = common.get_server_by_name(server_name)
    tisapph.set_wavelength_nm(800)
    print(tisapph.get_wavelength_nm())
    tisapph.set_wavelength_nm(800.1)
    print(tisapph.get_wavelength_nm())
    tisapph.set_wavelength_nm(800)
    print(tisapph.get_wavelength_nm())


def test_meas_etalon_timing():
    server_name_tisapph = "tisapph_M2_solstis"
    tisapph = common.get_server_by_name(server_name_tisapph)

    server_name_meter = "multimeter_KEIT_daq6510"
    meter = common.get_server_by_name(server_name_meter)

    tisapph.set_wavelength_nm(800)
    tisapph.tune_etalon_relative(50)  # Reset back to 50% voltage setting

    print("Etalon setting =", tisapph.get_etalon_tune_status())
    # return
    loop_start = time.time()
    for e in range(50, 90, 2):
        if e <= 60:
            start = time.time()
            tisapph.tune_etalon_relative(e)
            print("Tisapph tune time =", time.time() - start)

            start_meter = time.time()
            meter.read()
            print("Meter measurement time =", time.time() - start_meter)
        else:
            tisapph.tune_etalon_relative(e)
            meter.read()

    print("Total time elapsed =", time.time() - loop_start)
    tisapph.tune_etalon_relative(50)


def test_tisapph_etalon():
    server_name = "tisapph_M2_solstis"
    tisapph = common.get_server_by_name(server_name)
    tisapph.set_wavelength_nm(804)
    print(f"Initial: {tisapph.get_wavelength_nm()}")

    tisapph.tune_etalon_relative(50)
    print(f"Reset etalon to 0.5: {tisapph.get_wavelength_nm()}")
    print(tisapph.get_etalon_tune_status())

    tisapph.tune_etalon_relative(70)
    print(f"Tune etalon: {tisapph.get_wavelength_nm()}")
    print(tisapph.get_etalon_tune_status())


def test_measure_shutter_timing():
    server_name_STAN = "shutter_STAN_sr474"
    shutter = common.get_server_by_name(server_name_STAN)

    server_name_meter = "multimeter_KEIT_daq6510"
    meter = common.get_server_by_name(server_name_meter)

    start = time.time()
    shutter.close(1)
    print("Shutter close time =", time.time() - start)
    start_measure = time.time()
    meter.read()
    print("Meter measurement time =", time.time() - start_measure)
    shutter.open(1)


def test_multimeter_avg():
    server_name = "multimeter_KEIT_daq6510"
    meter = common.get_server_by_name(server_name)
    nplc = 1
    num_meas_to_avg = 15
    filter_type = "hybrid"
    meter.set_averaging_params(nplc, num_meas_to_avg, filter_type)

    start_time = time.time()
    print(meter.read())
    print(f"Time elapsed: {time.time() - start_time}")

    start_time = time.time()
    meter.set_avg_window_size(20)
    print(meter.read())
    print(f"Time elapsed: {time.time() - start_time}")

    meter.turn_off_averaging()
    start_time = time.time()
    print(meter.read())
    print(f"Time elapsed: {time.time() - start_time}")


# def test_pump():
#     server_name = "tisapph_pump_COHE_verdi"
#     pump = common.get_server_by_name(server_name)
#     pump.


if __name__ == "__main__":
    kpl.init_kplotlib()

    email_recipient = ["mccambria@berkeley.edu", "jennychen42@berkeley.edu"]
    do_email = False
    try:
        # test_shutter()
        # test_multimeter()
        # test_multimeter_avg()
        # test_multimeter_stats()
        # test_sig_gen()
        # start = time.time()
        # test_multimeter_duration()
        # print(time.time() - start)
        # test_tisapph()
        # test_tisapph_etalon()
        # test_meas_etalon_timing()
        # test_measure_shutter_timing()
        # do_singlet_search()
        # do_singlet_search_with_etalon()
        # do_singlet_search_with_etalon_1()
        # do_singlet_search_with_etalon_2()
        do_singlet_search_1043()
        # pass

    except Exception as exc:
        if do_email:
            tb.send_exception_email(email_to=email_recipient)
        raise exc

    finally:
        if do_email:
            msg = "Experiment complete!"
            tb.send_email(msg, email_to=email_recipient)

        print()
        print("Routine complete")

        # Maybe necessary to make sure we don't interrupt a sequence prematurely
        # tb.poll_safe_stop()

        # Make sure everything is reset
        tb.reset_cfm()
        cxn = common.labrad_connect()
        cxn.disconnect()
        tb.reset_safe_stop()
        plt.show(block=True)
