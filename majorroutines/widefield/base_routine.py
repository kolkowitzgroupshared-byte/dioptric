# -*- coding: utf-8 -*-
"""
Widefield base routine for multi-NV experiments.

Runs a run→step→rep→exp loop for many spatially resolved NVs, with optional charge-prep,
per-rep imaging + charge-state estimation (thresholding or MLE), and drift compensation
between runs. Returns a raw_data dict containing counts[exp,nv,run,step,rep] (+ optional
averaged images) and metadata; supports multi-channel microwave control and optional IQ.

Created: 2023-12-06 (mccambria)
Updated: 2026-01-05 (Saroj Chand)
"""

import logging
import time
import traceback
from random import shuffle

import numpy as np

from majorroutines import targeting
from utils import common, widefield
from utils import positioning as pos
from utils import tool_belt as tb
from utils.constants import ChargeStateEstimationMode, CoordsKey


def charge_prep_no_verification_skip_first_rep(
    rep_ind, nv_list, initial_states_list=None
):
    if rep_ind > 0:
        charge_prep_base(nv_list, initial_states_list)


def charge_prep_no_verification(rep_ind, nv_list, initial_states_list=None):
    charge_prep_base(nv_list, initial_states_list)


# def charge_prep_first_rep_only(rep_ind, nv_list, initial_states_list=None):
#     if rep_ind == 0:
#         charge_prep_loop(nv_list, initial_states_list)


def charge_prep_loop(rep_ind, nv_list, initial_states_list=None):
    charge_prep_base(
        nv_list,
        initial_states_list,
        targeted_polarization=True,
        verify_charge_states=True,
    )


def charge_prep_base(
    nv_list,
    initial_states_list=None,
    targeted_polarization=True,
    verify_charge_states=False,
):
    # Initial setup
    pulse_gen = tb.get_server_pulse_gen()
    num_nvs = len(nv_list)
    states_list = initial_states_list

    # Inner function for determining which NVs to target
    def assemble_charge_pol_target_list(states_list):
        if states_list is not None:
            charge_pol_target_list = [el is None or el == 0 for el in states_list]
        else:
            charge_pol_target_list = [True for ind in range(num_nvs)]
        # charge_pol_target_list = [not (el) for el in charge_pol_target_list]
        # charge_pol_target_list = [False for ind in range(num_nvs)]
        return charge_pol_target_list

    if verify_charge_states:
        max_num_attempts = 10
        # max_num_attempts = 1
        out_of_attempts = False
        attempt_ind = 0

        # Loop until we have a reason to stop
        while True:
            out_of_attempts = attempt_ind == max_num_attempts
            charge_pol_target_list = assemble_charge_pol_target_list(states_list)

            # Reasons to stop
            no_more_targets = True not in charge_pol_target_list
            charge_pol_complete = no_more_targets or out_of_attempts
            pulse_gen.insert_input_stream(
                "_cache_charge_pol_incomplete", not charge_pol_complete
            )
            if charge_pol_complete:
                break

            pulse_gen.insert_input_stream("_cache_target_list", charge_pol_target_list)

            _, _, states_list = read_and_process_image(nv_list)

            # Move on to next attempt
            attempt_ind += 1
    elif targeted_polarization:
        charge_pol_target_list = assemble_charge_pol_target_list(states_list)
        # print(charge_pol_target_list)
        pulse_gen.insert_input_stream("_cache_target_list", charge_pol_target_list)
    else:
        pass


def read_and_process_image(nv_list):
    camera = tb.get_server_camera()
    img_str = camera.read()
    img_array_adus, baseline = widefield.img_str_to_array(img_str)
    # baseline = 300
    img_array = widefield.adus_to_photons(img_array_adus, baseline=baseline)

    def get_counts(nv_sig):
        pixel_coords = pos.get_nv_coords(nv_sig, CoordsKey.PIXEL)
        return widefield.integrate_counts(img_array, pixel_coords)

    counts_list = [get_counts(nv) for nv in nv_list]

    config = common.get_config_dict()
    charge_state_estimation_mode = config["charge_state_estimation_mode"]
    if charge_state_estimation_mode == ChargeStateEstimationMode.THRESHOLDING:
        num_nvs = len(nv_list)
        states_list = []
        for nv_ind in range(num_nvs):
            states_list.append(
                tb.threshold(counts_list[nv_ind], nv_list[nv_ind].threshold)
            )
    elif charge_state_estimation_mode == ChargeStateEstimationMode.MLE:
        # start = time.time()
        states_list = widefield.charge_state_mle(nv_list, img_array)
        # stop = time.time()
        # print(stop - start)

    counts_list = np.array(counts_list)
    states_list = np.array(states_list)

    return img_array, counts_list, states_list


def dtype_clip(arr, dtype):
    min_val = np.finfo(dtype).min
    max_val = np.finfo(dtype).max
    arr = np.where(arr >= min_val, arr, min_val)
    arr = np.where(arr <= max_val, arr, max_val)
    return arr


def main(
    nv_list,
    num_steps,
    num_reps,
    num_runs,
    run_fn=None,
    step_fn=None,
    uwave_ind_list=[0, 1],
    uwave_freq_list=None,
    num_exps=2,
    ref_by_rep_parity=True,
    load_iq=False,
    save_images=False,
    save_images_avg_reps=True,
    save_images_downsample_factor=None,
    stream_load_in_run_fn=True,
    charge_prep_fn=None,
) -> dict:
    """Base routine for widefield experiments with many spatially resolved NV centers.

    The routine is broken down into a sequence of identical "runs". In between runs
    we update the sample drift so that all our targeting remains accurate. Each run
    consists of a sequence of "steps" where each step represents a specific value of
    the dependent variable (e.g. a specific frequency in an ESR experiment). In turn
    each step consists of some number of identical "repetitions", where in each
    repetition we may perform one or several distinct experiments (e.g. a signal and a
    reference experiment).

    Parameters
    ----------
    nv_list : list(nv_sig)
        List of NV sig objects to interrogate
    num_runs : int
        Number of runs
    num_steps : int
        Number of steps
    num_reps : int
        Number of repetitions, or "reps"
    step_fn : function, optional
        Function to run when moving to a new step. Most likely we want to load a new
        sequence onto the pulse streamer here, but we may also want to just update a
        microwave frequency or something similar. If None, do nothing. By default None
    uwave_ind : int, optional
        Index of microwave signal chain to use, by default 0
    uwave_freq : float, optional
        Microwave frequency to set in GHz, by default retrieved from config
    num_exps_per_rep : int, optional
        Number of experiments to perform in a single rep, by default 2
    load_iq : bool, optional
        Whether to load IQ modulation for the microwave signal chain, by default False
    save_images : bool, optional
        Whether to return the images from the experiments to the caller routine.
        To save space, reps are averaged over. Results in large files, roughly 1 gb
        per hour of runtime after compression. By default False

    Returns
    -------
    ndarray
        Array of photon counts. Indexing is experiment, nv, run, step, rep
    dict
        Raw data object to save. Populated with basic data, such as the counts array,
        nv_list, and images if save_images is True. Add whatever routine-specific
        data we should also save onto this dict before writing it to the cloud
    """
    ### Some initial setup

    tb.reset_cfm()

    repr_nv_sig = widefield.get_repr_nv_sig(nv_list)
    pos.set_xyz_on_nv(repr_nv_sig)
    # shuffle(nv_list)  # SBC ->
    num_nvs = len(nv_list)
    camera = tb.get_server_camera()
    pulse_gen = tb.get_server_pulse_gen()

    # Sig gen setup - all but turning on the output
    if isinstance(uwave_ind_list, int):
        uwave_ind_list = [uwave_ind_list]
    for ind in range(len(uwave_ind_list)):
        uwave_ind = uwave_ind_list[ind]
        uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
        uwave_power = uwave_dict["uwave_power"]
        freq = uwave_dict["frequency"]
        rabi_period = uwave_dict["rabi_period"]

        if uwave_freq_list is None:
            freq = uwave_dict["frequency"]
        else:
            freq = uwave_freq_list[uwave_ind]

        sig_gen = tb.get_server_sig_gen(uwave_ind)

        if load_iq:
            sig_gen.load_iq()
            # uwave_power += 0.4
            sig_gen.set_amp(uwave_power)
            sig_gen.set_freq(freq)
            print(
                f"Using IQ Modulation: Rabi frequency {freq} and Power={uwave_power} dBm"
            )
        else:
            sig_gen.set_amp(uwave_power)
            sig_gen.set_freq(freq)
            print(
                f"No IQ Modulation: Rabi frequency {freq} GHz and period {rabi_period}ns "
            )

    # Sig gen setup - all but turning on the output
    # if isinstance(uwave_ind_list, int):
    #     uwave_ind_list = [uwave_ind_list]

    # for ind, uwave_ind in enumerate(uwave_ind_list):
    #     uwave_dict = tb.get_virtual_sig_gen_dict(uwave_ind)
    #     uwave_power = uwave_dict["uwave_power"]
    #     rabi_period = uwave_dict.get("rabi_period", None)

    #     if uwave_freq_list is None:
    #         freq = uwave_dict["frequency"]
    #     else:
    #         freq = uwave_freq_list[uwave_ind]   # assumes uwave_freq_list indexed by uwave_ind

    #     sig_gen = tb.get_server_sig_gen(uwave_ind)

    #     # IQ only for first two SGs in uwave_ind_list
    #     do_iq = (ind < 2)

    #     if do_iq:
    #         sig_gen.load_iq()
    #         sig_gen.set_amp(uwave_power)
    #         sig_gen.set_freq(freq)
    #         print(f"[SG{uwave_ind}] IQ ON:  freq={freq}  amp={uwave_power} dBm")
    #     else:
    #         sig_gen.set_amp(uwave_power)
    #         sig_gen.set_freq(freq)
    #         print(f"[SG{uwave_ind}] IQ OFF: freq={freq}  rabi_period={rabi_period} ns  amp={uwave_power} dBm")

    ### Data tracking

    counts_dtype = np.float16
    counts = np.empty(
        (num_exps, num_nvs, num_runs, num_steps, num_reps), dtype=counts_dtype
    )
    pixel_drifts = np.empty((num_runs, 2))
    if save_images:
        shape = widefield.get_img_array_shape()
        if save_images_downsample_factor is not None:
            shape = [
                int(np.floor(shape[ind] / save_images_downsample_factor))
                for ind in range(2)
            ]
        save_images_num_reps = 1 if save_images_avg_reps else num_reps
        img_arrays_dtype = np.float16
        img_arrays = np.empty(
            (num_exps, num_runs, num_steps, save_images_num_reps, *shape),
            dtype=img_arrays_dtype,
        )
        if save_images_avg_reps:
            img_array_reps_cache = np.empty((num_exps, num_reps, *shape))
    step_ind_master_list = [None for ind in range(num_runs)]
    step_ind_list = list(range(0, num_steps))
    crash_counter = [None] * num_runs

    ### Collect the data

    try:
        # Runs loop
        for run_ind in range(num_runs):
            num_attempts = 15
            attempt_ind = 0

            while True:
                try:
                    print(f"\nRun index: {run_ind}")

                    states_list = None
                    first_step = True

                    for ind in uwave_ind_list:
                        sig_gen = tb.get_server_sig_gen(ind)
                        sig_gen.uwave_on()
                        # if load_iq:
                        #     sig_gen.load_iq()

                    shuffle(step_ind_list)
                    if run_fn is not None:
                        run_fn(step_ind_list)

                    camera.arm()

                    # Steps loop
                    for step_ind in step_ind_list:
                        if step_fn is not None:
                            step_fn(step_ind)

                        if first_step:
                            pulse_gen.stream_start()
                            first_step = False
                        else:
                            if stream_load_in_run_fn:
                                pulse_gen.resume()
                            else:
                                pulse_gen.stream_start()

                        # Reps loop
                        for rep_ind in range(num_reps):
                            last_rep = rep_ind == num_reps - 1
                            for exp_ind in range(num_exps):
                                last_exp = exp_ind == num_exps - 1
                                if charge_prep_fn is not None:
                                    charge_prep_fn(
                                        rep_ind,
                                        nv_list,
                                        initial_states_list=states_list,
                                    )
                                ret_vals = read_and_process_image(nv_list)
                                # print(f"rep_ind: {rep_ind}, step_ind: {step_ind}")
                                img_array, counts_list, states_list = ret_vals
                                counts[exp_ind, :, run_ind, step_ind, rep_ind] = (
                                    dtype_clip(counts_list, counts_dtype)
                                )

                                if save_images:
                                    if save_images_downsample_factor is not None:
                                        img_array = widefield.downsample_img_array(
                                            img_array, save_images_downsample_factor
                                        )

                                    if save_images_avg_reps:
                                        img_array_reps_cache[exp_ind, rep_ind] = (
                                            img_array
                                        )
                                    else:
                                        img_arrays[
                                            exp_ind, run_ind, step_ind, rep_ind
                                        ] = dtype_clip(img_array, img_arrays_dtype)

                                    # Process img_arrays_reps_cache if this is the last rep
                                    if save_images_avg_reps and last_rep:
                                        if ref_by_rep_parity and last_exp:
                                            img_arrays_to_avg = img_array_reps_cache[
                                                exp_ind, ::2
                                            ]
                                        else:
                                            img_arrays_to_avg = img_array_reps_cache[
                                                exp_ind
                                            ]
                                        avg_img_array = np.average(
                                            img_arrays_to_avg, axis=0
                                        )
                                        img_arrays[exp_ind, run_ind, step_ind, 0] = (
                                            dtype_clip(avg_img_array, img_arrays_dtype)
                                        )

                    ### Move on to the next run

                    # Turn stuff off
                    pulse_gen.halt()
                    camera.disarm()
                    for ind in uwave_ind_list:
                        sig_gen = tb.get_server_sig_gen(ind)
                        sig_gen.uwave_off()

                    # Record step order
                    step_ind_master_list[run_ind] = step_ind_list.copy()

                    # Update global coordinates (new)
                    targeting.compensate_for_drift(repr_nv_sig, no_crash=True)

                    crash_counter[run_ind] = attempt_ind

                    break

                except Exception as exc:
                    pulse_gen.halt()
                    # Camera disarmed automatically

                    nuvu_237 = "NuvuException: 237"
                    nuvu_214 = "NuvuException: 214"
                    if nuvu_237 in str(exc):
                        print(nuvu_237)
                    elif nuvu_214 in str(exc):
                        print(nuvu_214)
                    else:
                        print(traceback.format_exc())
                        raise exc

                    attempt_ind += 1
                    if attempt_ind == num_attempts:
                        raise RuntimeError("Maxed out number of attempts")

    except Exception:
        print(traceback.format_exc())

    ### Return

    raw_data = {
        "nv_list": nv_list,
        "num_steps": num_steps,
        "num_reps": num_reps,
        "num_runs": num_runs,
        "uwave_ind": uwave_ind_list,
        "uwave_freq": uwave_freq_list,
        "num_exps_per_rep": num_exps,
        "load_iq": load_iq,
        "step_ind_master_list": step_ind_master_list,
        "counts-units": "photons",
        "counts": counts,
        "pixel_drifts": pixel_drifts,
        "crash_counter": crash_counter,
    }
    if save_images:
        raw_data |= {
            "img_arrays-units": "photons",
            "img_arrays": img_arrays,
        }
    return raw_data


if __name__ == "__main__":
    pass
