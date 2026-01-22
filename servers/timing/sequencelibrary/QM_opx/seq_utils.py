# -*- coding: utf-8 -*-
"""
QM OPX sequence utils. Should only be used by sequence files

Created June 25th, 2023

@author: mccambria

@author: sbchand
"""

import logging
import time
from functools import cache

import numpy as np
from qm import qua

from utils import common
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ModMode, VirtualLaserKey

# region Public QUA macros - sequence management


def init(num_nvs=None):
    """This should be the first command we call in any sequence"""

    # Declare cached QUA variables (helps reduce compile times)
    global _cache_x_freq
    _cache_x_freq = qua.declare(int)
    global _cache_y_freq
    _cache_y_freq = qua.declare(int)
    global _cache_x_freq_2
    _cache_x_freq_2 = qua.declare(int)
    global _cache_y_freq_2
    _cache_y_freq_2 = qua.declare(int)

    global _cache_duration
    _cache_duration = qua.declare(int)

    global _cache_amp
    _cache_amp = qua.declare(qua.fixed)

    global _cache_target
    _cache_target = qua.declare(bool)

    global _cache_macro_run_aods
    _cache_macro_run_aods = {}
    macro_run_aods()

    global _cache_pol_reps_ind
    _cache_pol_reps_ind = qua.declare(int)

    global _cache_charge_pol_incomplete
    _cache_charge_pol_incomplete = qua.declare_input_stream(
        bool, name="_cache_charge_pol_incomplete"
    )

    if num_nvs is not None:
        global _cache_target_list
        _cache_target_list = qua.declare_input_stream(
            bool, name="_cache_target_list", size=num_nvs
        )


def handle_reps(one_rep_macro, num_reps: int, wait_for_trigger: bool = True):
    """Handle repetitions of a given sequence - you just have to pass
    a function defining the behavior for a single loop. Optionally
    waits for trigger pulse between loops.


    Parameters
    ----------
    one_rep_macro : qua.macro
        QUA macro to be repeated
    num_reps : int
        Number of times to repeat, -1 for infinite loop
    wait_for_trigger : bool, optional
        Whether or not to pause execution between loops until a trigger
        pulse is received by the OPX, by default True
    """
    if num_reps == -1:
        with qua.infinite_loop_():
            one_rep_macro()
            if wait_for_trigger:
                macro_wait_for_trigger()
    elif num_reps == 1:
        one_rep_macro()
    else:
        handle_reps_ind = qua.declare(int)
        with qua.for_(
            handle_reps_ind, 0, handle_reps_ind < num_reps, handle_reps_ind + 1
        ):
            one_rep_macro(handle_reps_ind)
            if wait_for_trigger:
                macro_wait_for_trigger()


def macro_wait_for_trigger():
    """Pauses the entire sequence and waits for a trigger pulse from the camera.
    The wait does not start until all running pulses finish
    """
    # Wait_for_trigger requires us to pass some element so use an arbitrary dummy
    dummy_element = "do_camera_trigger"
    qua.align()
    qua.wait_for_trigger(dummy_element)


def macro_pause():
    """Pause the sequence and pass control back to the top-level routine for processing,
    switching signal generator frequencies, etc.
    """
    buffer = get_widefield_operation_buffer()
    # Make sure everything is off before pausing for the next step
    qua.align()
    qua.wait(buffer)
    qua.pause()


# endregion


# region Public QUA macros - laser and microwave pulses
def macro_polarize(
    coords_list: list[list[float]],
    duration_list: list[int] = None,
    amp_list: list[float] = None,
    duration_override: int = None,
    amp_override: float = None,
    targeted_polarization: bool = False,
    verify_charge_states: bool = False,
    spin_pol: bool = True,
    spin_pol_duration_override: int = None,
    spin_pol_amp_override: float = None,
    aod_accees_time_override: int = None
):
    """Apply a green polarization pulse to each coordinate pair in the passed coords_list.
    Supports conditional charge-state initialization with targeted_polarization and
    verify_charge_states parameters. (See base_routine for details.) Optionally apply
    a yellow widefield spin polarization pulse after green serial charge polarization

    Parameters
    ----------
    coords_list : list[list[float]]
        List of coordinate pairs to target
    duration_list : list[int], optional
        List of pulse durations, by default whatever value is in config
    amp_list : list[float], optional
        List of pulse amplitudes, by default whatever value is in config
    duration_override : int, optional
        Pulse duration for all pulses - overrides duration_list.
        Useful for parameters sweeps. By default do not override
    amp_override : float, optional
        Pulse amplitude for all pulses - overrides amp_list.
        Useful for parameters sweeps. By default do not override
    targeted_polarization : bool, optional
        Whether or not to apply charge polarization to only certain NVs. Requires setting
        _cache_target_list on the client via insert_input_stream call. (See base_routine
        for details.) By default False
    verify_charge_states : bool, optional
        Whether or not to verify charge states of NVs for high-fidelity conditional charge-
        state initialization. Requires setting _cache_charge_pol_incomplete on the client
        via insert_input_stream call. (See base_routine for details.) By default False
    spin_pol : bool, optional
        Whether or not to apply a yellow widefield spin polarization pulse
        after green serial charge polarization. By default True
    """

    global _cache_charge_pol_incomplete
    global _cache_target_list

    pol_laser_name = tb.get_physical_laser_name(VirtualLaserKey.CHARGE_POL)
    pulse_name = "charge_pol"
    macro_run_aods(laser_names=[pol_laser_name], aod_suffices=[pulse_name])

    def charge_pol_sub():
        do_target_list = _cache_target_list if targeted_polarization else None
        _macro_pulse_series(
            pol_laser_name,
            pulse_name,
            coords_list,
            duration_list,
            amp_list,
            duration_override,
            amp_override,
            aod_accees_time_override,
            do_target_list,
        )

    if verify_charge_states:
        qua.advance_input_stream(_cache_charge_pol_incomplete)
        with qua.while_(_cache_charge_pol_incomplete):
            qua.advance_input_stream(_cache_target_list)
            charge_pol_sub()
            macro_charge_state_readout()
            macro_wait_for_trigger()
            qua.advance_input_stream(_cache_charge_pol_incomplete)
    elif targeted_polarization:
        qua.advance_input_stream(_cache_target_list)
        charge_pol_sub()
    else:
        charge_pol_sub()

    ### SBC: updated Spin polarization which includes sweeping amp and duration
    if spin_pol:
        macro_spin_polarize(spin_pol_duration_override, spin_pol_amp_override)

    ## previous version of spin_pol
    # if spin_pol:
    #     spin_pol_laser_name = tb.get_physical_laser_name(
    #         VirtualLaserKey.WIDEFIELD_SPIN_POL
    #     )
    #     spin_pol_laser_el = get_laser_mod_element(spin_pol_laser_name)
    #     buffer = get_widefield_operation_buffer()
    #     qua.align()
    #     qua.play("spin_pol", spin_pol_laser_el)
    #     qua.wait(buffer, spin_pol_laser_el)


# SBC: spin polarization updates
def macro_spin_polarize(duration: int = None, amp: float = None):
    """
    Apply a widefield yellow spin polarization pulse.

    Parameters
    ----------
    duration : int, optional
        Pulse duration in clock cycles (cc). Defaults to config value.
    amp : float, optional
        Pulse amplitude. Defaults to config value.
    """
    spin_pol_laser_name = tb.get_physical_laser_name(VirtualLaserKey.WIDEFIELD_SPIN_POL)
    spin_pol_laser_el = get_laser_mod_element(spin_pol_laser_name)

    if duration is None:
        duration = get_default_spin_pol_duration()

    buffer = get_widefield_operation_buffer()

    qua.align()
    if amp is None:
        qua.play("spin_pol", spin_pol_laser_el, duration=duration)
    else:
        qua.play("spin_pol" * qua.amp(amp), spin_pol_laser_el, duration=duration)

    qua.wait(buffer, spin_pol_laser_el)


def macro_ionize(ion_coords_list: list[list[float]], do_target_list: list[bool] = None):
    """Apply an ionization pulse to each coordinate pair in the passed coords_list.

    Parameters
    ----------
    ion_coords_list : list[list[float]]
        List of coordinate pairs to target
    do_target_list : list[bool], optional
        List of whether to target an NV or not. Used to skip certain NVs.
        Default value None targets all NVs
    """
    ion_laser_name = tb.get_physical_laser_name(VirtualLaserKey.ION)
    ion_pulse_name = "ion"
    macro_run_aods([ion_laser_name], aod_suffices=[ion_pulse_name])
    _macro_pulse_series(
        ion_laser_name, ion_pulse_name, ion_coords_list, do_target_list=do_target_list
    )


def macro_scc(
    scc_coords_list: list[list[float]],
    scc_duration_list: list[int] = None,
    scc_amp_list: list[float] = None,
    scc_duration_override: int = None,
    scc_amp_override: float = None,
    aod_accees_time_override: int = None,
    do_target_list: list[bool] = None,
):
    """Apply an SCC pulse to each coordinate pair in the passed coords_list.
    Checks config for whether or not to include a shelving pulse


    Parameters
    ----------
    scc_coords_list : list[list[float]]
        List of coordinate pairs to target for the SCC pulse itself
    scc_duration_list : list[int], optional
        List of pulse durations for the SCC pulse itself, by default whatever value is in config
    scc_amp_list : list[float], optional
        List of pulse amplitudes for the SCC pulse itself, by default whatever value is in config
    scc_duration_override : int, optional
        Pulse duration for all pulses for the SCC pulse itself - overrides duration_list.
        Useful for parameters sweeps. By default do not override
    scc_amp_override : float, optional
        Pulse amplitude for all pulses for the SCC pulse itself - overrides amp_list.
        Useful for parameters sweeps. By default do not override
    do_target_list : list[bool], optional
        List of whether to target an NV or not. Used to skip certain NVs.
        Default value None targets all NVs
    """

    config = common.get_config_dict()
    do_shelving_pulse = config["Optics"]["PulseSettings"]["scc_shelving_pulse"]

    if do_shelving_pulse:
        raise NotImplementedError()
        # if spin_flip_ind_list is not None:
        #     msg = "Shelving SCC with spin_flips not yet implemented."
        #     raise NotImplementedError(msg)
        # _macro_scc_shelving(
        #     scc_coords_list,
        #     scc_duration_list,
        #     shelving_coords_list,
        # )
    else:
        _macro_scc_no_shelving(
            scc_coords_list,
            scc_duration_list,
            scc_amp_list,
            scc_duration_override,
            scc_amp_override,
            aod_accees_time_override,
            do_target_list,
        )


def macro_charge_state_readout(duration: int = None, amp: float = None):
    """
    Pulse yellow to read out NV charge states in parallel.

    Parameters
    ----------
    duration : int or QuaVariable, optional
        Readout and pulse duration in clock cycles (cc). Defaults to whatever is in config.
    amp : float, optional
        Pulse amplitude. Defaults to whatever is in config.
    """
    readout_laser_name = tb.get_physical_laser_name(
        VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    )
    readout_laser_el = get_laser_mod_element(readout_laser_name, sticky=True)
    camera_el = "do_camera_trigger"

    # Handle static vs dynamic duration
    if duration is None:
        duration = get_default_charge_readout_duration()

    print(amp, duration)
    qua.align()
    if amp is not None:
        qua.play("charge_readout" * qua.amp(amp), readout_laser_el)
    else:
        qua.play("charge_readout", readout_laser_el)

    qua.play("on", camera_el)

    # Wait for the total readout duration
    # qua.wait(duration, readout_laser_el)
    # qua.wait(duration, camera_el)
    with qua.if_(duration < int(60e6 / 4)):
        qua.wait(duration, readout_laser_el)
        qua.wait(duration, camera_el)
    with qua.else_():
        qtr_duration = qua.declare(int)
        qua.assign(qtr_duration, duration // 4)
        wait_ind = qua.declare(int)
        with qua.for_(wait_ind, 0, wait_ind < 4, wait_ind + 1):
            qua.wait(qtr_duration, readout_laser_el)
            qua.wait(qtr_duration, camera_el)

    # Ramp down to zero
    qua.ramp_to_zero(readout_laser_el)
    qua.ramp_to_zero(camera_el)


def macro_pi_pulse(uwave_ind_list, duration_cc=None, phase=None, amp=1.0):
    _macro_uwave_pulse(
        uwave_ind_list,
        pulse_name="pi_pulse",
        duration_cc=duration_cc,
        phase=phase,
        amp=amp,
    )


def macro_pi_on_2_pulse(uwave_ind_list, duration_cc=None, phase=None, amp=1.0):
    _macro_uwave_pulse(
        uwave_ind_list,
        pulse_name="pi_on_2_pulse",
        duration_cc=duration_cc,
        phase=phase,
        amp=amp,
    )


def _get_iq_pulse(pulse_name, phase):
    return f"{pulse_name}_{phase}"


def _macro_uwave_pulse(
    uwave_ind_list, pulse_name="pi_pulse", duration_cc=None, phase=None, amp=1.0
):
    if uwave_ind_list is None:
        return

    for uwave_ind in uwave_ind_list:
        ## Setup

        elements = []  # List of elements controlling this specific microwave tone

        # Phase component calculations
        if phase is not None:
            i_el = get_sig_gen_i_element(uwave_ind)
            q_el = get_sig_gen_q_element(uwave_ind)
            elements.extend([i_el, q_el])
            iq_pulse = _get_iq_pulse(pulse_name, phase)

        sig_gen_el = get_sig_gen_element(uwave_ind)
        elements.append(sig_gen_el)

        qua.align(*elements)

        def pulse_sub(duration_arg=None):
            if phase is not None:
                qua.play(iq_pulse, i_el, duration=duration_arg)
                qua.play(iq_pulse, q_el, duration=duration_arg)
            qua.play(pulse_name, sig_gen_el, duration=duration_arg)

        if duration_cc is None:
            pulse_sub()
        else:
            with qua.if_(duration_cc > 0):
                pulse_sub(duration_cc)

        qua.align(*elements)


def macro_cpdd_drive(
    uwave_ind_list,
    phase_list,
    duration_cc,
    total_duration,
    pulse_name="pi_pulse",
):
    """
    Play a continuous drive with discrete phase changes (for CPDD).

    Args:
        uwave_ind_list: List of indices of microwave channels.
        phase_list: List of phases (degrees) to cycle through.
        duration_cc: Time (in clock cycles) between phase changes.
        amp: Amplitude.
    """
    if uwave_ind_list is None:
        return

    for uwave_ind in uwave_ind_list:
        i_el = get_sig_gen_i_element(uwave_ind)
        q_el = get_sig_gen_q_element(uwave_ind)
        sig_gen_el = get_sig_gen_element(uwave_ind)
        el_list = [i_el, q_el, sig_gen_el]

        qua.align(*el_list)
        # Turn on analog microwave output (switch) for full duration
        # with qua.strict_timing_():
        qua.play(pulse_name, sig_gen_el, duration=total_duration)
        for phase in phase_list:
            iq_pulse = _get_iq_pulse(pulse_name, phase=phase)
            qua.play(iq_pulse, i_el, duration=duration_cc)
            qua.play(iq_pulse, q_el, duration=duration_cc)

        qua.align(*el_list)


def macro_run_aods(
    laser_names: list[str] = None,
    aod_suffices: list[str] = None,
    amps: list[float] = None,
):
    """Turn on the AODs. They'll run indefinitely. Use a pulse_suffix to run a different
    named pulse from config (one with a different power) or pass amps to modulate the
    power manually

    Parameters
    ----------
    laser_names : list[str], optional
        Lasers whose AODs we'll turn on, by default all lasers with AODs
    aod_suffices : list[str], optional
        Pulse format is "aod_cw-<suffix>", by default no suffix
    amps : list[float], optional
        Amplitudes for the RF tone pulses. Amplitudes are multiplicative wrt the
        pulse default voltage, so an amp of 1.0 does nothing, by default None
    """
    # By default search the config for the lasers with AOD
    if laser_names is None:
        config = common.get_config_dict("purcell")
        positioners_dict = config["Positioning"]["Positioners"]
        positioners_keys = positioners_dict.keys()
        laser_names = []
        for key in positioners_keys:
            positioner = positioners_dict[key]
            if "aod" in positioner and positioner["aod"]:
                virtual_laser_key = positioner["opti_virtual_laser_key"]
                physical_laser_name = tb.get_physical_laser_name(virtual_laser_key)
                laser_names.append(physical_laser_name)

    num_lasers = len(laser_names)
    # Adjust the pulse names for the passed suffices
    base_pulse_name = "aod_cw"
    pulse_names = []
    if aod_suffices is None:
        aod_suffices = [None] * num_lasers
    pulse_names = []
    for ind in range(num_lasers):
        suffix = aod_suffices[ind]
        if suffix is None:
            pulse_name = base_pulse_name
        else:
            pulse_name = f"{base_pulse_name}-{suffix}"
        pulse_names.append(pulse_name)

    if amps is None:
        # amps = [None] * num_lasers
        amps = [1.0] * num_lasers

    ### Check if the requested pulse is already running using the cache

    global _cache_macro_run_aods
    skip_inds = []
    for ind in range(num_lasers):
        laser_name = laser_names[ind]
        aod_suffix = aod_suffices[ind]
        amp = amps[ind]
        # Don't cache commands with amps since those may be QUA variables and so
        # could behave unexpectedly
        if amp is not None:
            continue
        if laser_name in _cache_macro_run_aods:
            cache_dict = _cache_macro_run_aods[laser_name]
            if cache_dict["aod_suffix"] == aod_suffix:
                skip_inds.append(ind)
                continue
        # If we get here then the pulse is not already running, so cache it
        _cache_macro_run_aods[laser_name] = {}
        cache_dict = _cache_macro_run_aods[laser_name]
        cache_dict["aod_suffix"] = aod_suffix

    ### Actual commands here

    qua.align()

    for ind in range(num_lasers):
        if ind in skip_inds:
            continue
        laser_name = laser_names[ind]
        x_el = f"ao_{laser_name}_x"
        y_el = f"ao_{laser_name}_y"
        pulse_name = pulse_names[ind]

        qua.ramp_to_zero(x_el)
        qua.ramp_to_zero(y_el)

        amp = amps[ind]
        if amp is not None:
            qua.play(pulse_name * qua.amp(amp), x_el)
            qua.play(pulse_name * qua.amp(amp), y_el)
        else:
            qua.play(pulse_name, x_el)
            qua.play(pulse_name, y_el)


def macro_single_pulse(
    laser_name: str,
    coords: list[float],
    pulse_name: str,
    duration: int = None,
    amp: float = None,
    aod_accees_time: int = None,
    convert_to_Hz: bool = True,
):
    """Apply a single laser pulse at the passed coordinate pair

    Parameters
    ----------
    laser_name : str
        Name of laser to pulse
    coords : list[float]
        Coordinate pair to target
    pulse_name : str
        Name of the pulse to play
    duration : int, optional
        Pulse duration in cc, by default whatever is in config
    amp : float, optional
        Pulse amplitude, by default whatever is in config
    convert_to_Hz : bool, optional
        Whether to convert coords from MHz to Hz, by default True
    """
    qua.align()
    _macro_single_pulse(laser_name, coords, pulse_name, duration, amp, aod_accees_time, convert_to_Hz)


def macro_pulse_combo(
    laser_name_list: list[str],
    coords_list: list[float],
    pulse_name_list: list[str],
    delays: list[int],
    duration_list: list[int] = None,
    amp_list: list[float] = None,
    convert_to_Hz: bool = True,
):
    """Apply a combination of laser pulse at one point in space described by the passed
    coordinate pairs. Each position in a list here describes a different pulse

    Parameters
    ----------
    laser_name_list : list[str]
        Names of lasers to pulse
    coords_list : list[float]
        Coordinate pairs describing the target
    pulse_name_list : list[str]
        Names of the pulses to play
    delays : list[int]
        Delays for the pulses from the beginning of the combo. Allows pulses to be
        offset from one another
    duration_list : list[int], optional
        Pulse durations in ns, by default whatever is in config
    amp_list : list[float], optional
        Pulse amplitude, by default whatever is in config
    convert_to_Hz : bool, optional
        Whether to convert coords from MHz to Hz, by default True
    """
    num_pulses = len(laser_name_list)
    if delays is None:
        delays = [0 for ind in range(num_pulses)]
    if duration_list is None:
        duration_list = [None for ind in range(num_pulses)]

    qua.align()
    for ind in range(num_pulses):
        laser_name = laser_name_list[ind]
        delay = delays[ind]
        laser_el = get_laser_mod_element(laser_name)
        qua.wait(delay, laser_el)

        _macro_single_pulse(
            laser_name,
            coords_list[ind],
            pulse_name_list[ind],
            duration_list[ind],
            amp_list[ind],
            convert_to_Hz,
        )


# endregion


# region Private QUA macros
def _macro_single_pulse(
    laser_name: str,
    coords: list[float],
    pulse_name: str,
    duration: int = None,
    amp: float = None,
    aod_accees_time: int = None,
    convert_to_Hz: bool = True,
    
):
    """Apply a single laser pulse at the passed coordinate pair. Does not align
    before beginning the macro

    Parameters
    ----------
    laser_name : str
        Name of laser to pulse
    coords : list[float]
        Coordinate pair to target
    pulse_name : str
        Name of the pulse to play
    duration : int, optional
        Pulse duration in cc, by default whatever is in config
    amp : float, optional
        Pulse amplitude, by default whatever is in config
    convert_to_Hz : bool, optional
        Whether to convert coords from MHz to Hz, by default True
    """
    # Setup
    laser_el = get_laser_mod_element(laser_name)
    x_el = f"ao_{laser_name}_x"
    y_el = f"ao_{laser_name}_y"

    global _cache_pol_reps_ind

    buffer = get_widefield_operation_buffer()
    if aod_accees_time is not None:
        accees_time = aod_accees_time
    else:
        accees_time = get_aod_access_time()
        

    if convert_to_Hz:
        coords = [int(el * 10**6) for el in coords]

    if amp is None:
        qua.play("continue", x_el)
        qua.play("continue", y_el)
    else:
        with qua.if_(amp == 0):
            qua.play("continue", x_el)
            qua.play("continue", y_el)
        with qua.else_():
            macro_run_aods(
                laser_names=[laser_name], aod_suffices=[pulse_name], amps=[amp]
            )
    qua.update_frequency(x_el, coords[0])
    qua.update_frequency(y_el, coords[1])

    # Pulse the laser
    qua.wait(accees_time + buffer, laser_el)
    if duration is None:
        qua.play(pulse_name, laser_el)
    elif isinstance(duration, int) and duration == 0:
        pass
    else:
        qua.play(pulse_name, laser_el, duration=duration)
    qua.wait(buffer, laser_el)


def _macro_pulse_series(
    laser_name: str,
    pulse_name: str,
    coords_list: list[list[float]],
    duration_list: list[int] = None,
    amp_list: list[float] = None,
    duration_override: int = None,
    amp_override: float = None,
    aod_accees_time_override: int = None,
    do_target_list: list[bool] = None,
):
    """Apply a laser pulse to each coordinate pair in the passed coords_list.
    Pulses are applied in series from one location to the next.

    Parameters
    ----------
    laser_name : str
        Name of laser to pulse
    pulse_name : str
        Name of the pulse to play
    coords_list : list[list[float]]
        List of coordinate pairs to target
    duration_list : list[int], optional
        List of pulse durations, by default whatever value is in config
    amp_list : list[float], optional
        List of pulse amplitudes, by default whatever value is in config
    duration_override : int, optional
        Pulse duration for all pulses - overrides duration_list.
        Useful for parameters sweeps. By default do not override
    amp_override : float, optional
        Pulse amplitude for all pulses - overrides amp_list.
        Useful for parameters sweeps. By default do not override
    do_target_list : list[bool], optional
        List of whether to target an NV or not. Used to skip certain NVs.
        Default value None targets all NVs
    """
    if len(coords_list) == 0:
        return

    # Unpack the coords and convert to Hz
    x_coords_list = [int(el[0] * 10**6) for el in coords_list]
    y_coords_list = [int(el[1] * 10**6) for el in coords_list]

    # Convert durations to clock cycles
    if duration_list is not None:
        duration_list = [convert_ns_to_cc(el) for el in duration_list]

    # These are declared in init
    global _cache_x_freq, _cache_y_freq, _cache_duration, _cache_amp, _cache_target

    def macro_sub():
        if duration_override is not None:
            duration = duration_override
        elif duration_list is not None:
            duration = _cache_duration
        else:
            duration = None
        if amp_override is not None:
            amp = amp_override
        elif amp_list is not None:
            amp = _cache_amp
        else:
            amp = None
            
        if aod_accees_time_override is not None:
            aod_accees_time = aod_accees_time_override
        else:
            aod_accees_time= None
            
        with qua.if_(_cache_target):
            macro_single_pulse(
                laser_name,
                (_cache_x_freq, _cache_y_freq),
                pulse_name=pulse_name,
                duration=duration,
                amp=amp,
                aod_accees_time=aod_accees_time,
                convert_to_Hz=False,
            )

    qua.align()

    # Adjust the for each based on what lists are populated
    list_1 = [_cache_x_freq, _cache_y_freq]
    list_2 = [x_coords_list, y_coords_list]
    if duration_list is not None:
        list_1.append(_cache_duration)
        list_2.append(duration_list)
    if amp_list is not None:
        list_1.append(_cache_amp)
        list_2.append(amp_list)
    if do_target_list is not None:
        list_1.append(_cache_target)
        list_2.append(do_target_list)
    else:  # Just set _cache_target to true if we want to hit every NV
        qua.assign(_cache_target, True)

    with qua.for_each_(tuple(list_1), tuple(list_2)):
        macro_sub()


def _macro_scc_shelving(
    scc_coords_list,
    scc_duration_list,
    spin_flip_ind_list,
    uwave_ind_list,
    shelving_coords_list,
    exp_spin_flip=True,
):
    """Needs work"""
    shelving_laser_name = tb.get_physical_laser_name(VirtualLaserKey.SHELVING)
    ion_laser_name = tb.get_physical_laser_name(VirtualLaserKey.SCC)
    laser_name_list = [shelving_laser_name, ion_laser_name]
    shelving_pulse_name = "shelving"
    ion_pulse_name = "scc"
    pulse_name_list = [shelving_pulse_name, ion_pulse_name]
    shelving_laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.SHELVING)
    shelving_pulse_duration = shelving_laser_dict["duration"]
    shelving_scc_gap_ns = 0
    scc_delay = convert_ns_to_cc(shelving_pulse_duration + shelving_scc_gap_ns)
    delays = [0, scc_delay]
    # duration_list = [None, ion_duration]

    macro_run_aods(laser_name_list, aod_suffices=pulse_name_list)

    # Unpack the coords and convert to Hz
    x_shelving_coords_list = [int(el[0] * 10**6) for el in shelving_coords_list]
    y_shelving_coords_list = [int(el[1] * 10**6) for el in shelving_coords_list]
    x_scc_coords_list = [int(el[0] * 10**6) for el in scc_coords_list]
    y_scc_coords_list = [int(el[1] * 10**6) for el in scc_coords_list]

    # These are declared in macro_run_aods
    global _cache_x_freq
    global _cache_y_freq
    global _cache_x_freq_2
    global _cache_y_freq_2
    freq_vars = (_cache_x_freq, _cache_y_freq, _cache_x_freq_2, _cache_y_freq_2)
    freq_lists = (
        x_shelving_coords_list,
        y_shelving_coords_list,
        x_scc_coords_list,
        y_scc_coords_list,
    )

    qua.align()
    with qua.for_each_(freq_vars, freq_lists):
        macro_pulse_combo(
            laser_name_list,
            ((_cache_x_freq, _cache_y_freq), (_cache_x_freq_2, _cache_y_freq_2)),
            pulse_name_list,
            # duration_list=duration_list,
            delays=delays,
            convert_to_Hz=False,
        )


def _macro_scc_no_shelving(
    coords_list: list[list[float]],
    duration_list: list[int] = None,
    amp_list: list[float] = None,
    duration_override: int = None,
    amp_override: float = None,
    aod_accees_time_override: int = None,
    do_target_list: list[bool] = None,
):
    """Perform spin-to-charge conversion (SCC) on NVs in series without a shelving pulse

    Parameters
    ----------
    coords_list : list[list[float]]
        List of coordinate pairs to target
    duration_list : list[int], optional
        List of pulse durations, by default whatever value is in config
    amp_list : list[float], optional
        List of pulse amplitudes, by default whatever value is in config
    duration_override : int, optional
        Pulse duration for all pulses - overrides duration_list.
        Useful for parameters sweeps. By default do not override
    amp_override : float, optional
        Pulse amplitude for all pulses - overrides amp_list.
        Useful for parameters sweeps. By default do not override
    do_target_list : list[bool], optional
        List of whether to target an NV or not. Used to skip certain NVs.
        Default value None targets all NVs
    """
    # Basic setup

    scc_laser_name = tb.get_physical_laser_name(VirtualLaserKey.SCC)
    scc_pulse_name = "scc"
    macro_run_aods([scc_laser_name], aod_suffices=[scc_pulse_name])

    # Actual commands

    # MCC antiphase by orientation
    # if exp_spin_flip:
    #     macro_pi_pulse(uwave_ind_list[:1])

    _macro_pulse_series(
        scc_laser_name,
        scc_pulse_name,
        coords_list,
        duration_list,
        amp_list,
        duration_override,
        amp_override,
        aod_accees_time_override,
        do_target_list,
    )


# endregion
# region Getters and utils


def convert_ns_to_cc(duration_ns, allow_rounding=False, allow_zero=False):
    """Convert a duration from nanoseconds to clock cycles"""
    if duration_ns is None:
        return None
    if not allow_rounding and duration_ns % 4 != 0:
        raise RuntimeError("OPX pulse durations (in ns) must be divisible by 4")
    if not allow_zero and duration_ns == 0:
        raise RuntimeError("OPX pulse duration 0 not supported here")
    if 0 < duration_ns < 16:
        raise RuntimeError("Minimum OPX pulse duration is 16 ns")
    return round(duration_ns / 4)


@cache
def get_pi_pulse_duration(uwave_ind):
    """Returns the duration of a pi pulse on the specified microwave
    channel in units of clock cycles. Useful for adjusting interpulse
    wait times to account for finite pulse duration effects.
    """
    return convert_ns_to_cc(get_rabi_period(uwave_ind) // 2)


@cache
def get_pi_on_2_pulse_duration(uwave_ind):
    """Returns the duration of a pi/2 pulse on the specified microwave
    channel in units of clock cycles. Useful for adjusting interpulse
    wait times to account for finite pulse duration effects.
    """
    return convert_ns_to_cc(get_rabi_period(uwave_ind) // 4)


@cache
def get_default_charge_readout_duration():
    readout_laser_dict = tb.get_virtual_laser_dict(
        VirtualLaserKey.WIDEFIELD_CHARGE_READOUT
    )
    readout_duration_ns = readout_laser_dict["duration"]
    return convert_ns_to_cc(readout_duration_ns)


@cache
def get_default_spin_pol_duration():
    spin_pol_laser_dict = tb.get_virtual_laser_dict(VirtualLaserKey.WIDEFIELD_SPIN_POL)
    spin_pol_duration_ns = spin_pol_laser_dict["duration"]
    return convert_ns_to_cc(spin_pol_duration_ns)


@cache
def get_default_pulse_duration():
    """Get the default OPX pulse duration in units of clock cycles"""
    return get_common_duration_cc("default_pulse_duration")


@cache
def get_aod_access_time():
    return get_common_duration_cc("aod_access_time")


@cache
def get_widefield_operation_buffer():
    return get_common_duration_cc("widefield_operation_buffer")


@cache
def get_uwave_buffer():
    return get_common_duration_cc("uwave_buffer")


@cache
def get_iq_buffer():
    return get_common_duration_cc("iq_buffer")


@cache
def get_common_duration_cc(key):
    common_duration_ns = tb.get_common_duration(key)
    common_duration_cc = convert_ns_to_cc(common_duration_ns)
    return common_duration_cc


@cache
def get_laser_mod_element(laser_name, sticky=False):
    physical_laser_dict = tb.get_physical_laser_dict(laser_name)
    mod_mode = physical_laser_dict["mod_mode"]
    if sticky:
        if mod_mode == ModMode.ANALOG:
            laser_mod_element = f"ao_{laser_name}_am_sticky"
        elif mod_mode == ModMode.DIGITAL:
            laser_mod_element = f"do_{laser_name}_dm_sticky"
    else:
        if mod_mode == ModMode.ANALOG:
            laser_mod_element = f"ao_{laser_name}_am"
        elif mod_mode == ModMode.DIGITAL:
            laser_mod_element = f"do_{laser_name}_dm"
    return laser_mod_element


@cache
def get_sig_gen_element(uwave_ind=0):
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = virtual_sig_gen_dict["physical_name"]
    sig_gen_element = f"do_{sig_gen_name}_dm"
    return sig_gen_element


@cache
def get_sig_gen_i_element(uwave_ind=0):
    """Returns the IQ-modulated signal generator element for a given uwave index."""
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = virtual_sig_gen_dict["physical_name"]
    sig_gen_element = f"ao_{sig_gen_name}_i"
    return sig_gen_element


@cache
def get_sig_gen_q_element(uwave_ind=0):
    """Returns the IQ-modulated signal generator element for a given uwave index."""
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    sig_gen_name = virtual_sig_gen_dict["physical_name"]
    sig_gen_element = f"ao_{sig_gen_name}_q"
    return sig_gen_element


@cache
def get_rabi_period(uwave_ind=0):
    virtual_sig_gen_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    rabi_period_ns = virtual_sig_gen_dict["rabi_period"]
    rabi_period = convert_ns_to_cc(rabi_period_ns)
    return rabi_period


# endregion

if __name__ == "__main__":
    print(get_default_charge_readout_duration())
    print(get_sig_gen_i_element(uwave_ind=2))
