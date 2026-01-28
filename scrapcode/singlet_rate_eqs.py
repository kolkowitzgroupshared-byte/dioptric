# -*- coding: utf-8 -*-
"""
Plot SNR as a function of green and NIR excitation rates for a CW experiment looking to
determine the wavelength of the ground triplet to excited singlet transition

Created on July 31st, 2025

@author: mccambria
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig

from utils import kplotlib as kpl

# Excited to ground state radiative lifetime is around 12 ns
exc_decay_rate = 1 / 12
# Singlet to ground state non-radiative lifetime is 200 ns
singlet_decay_rate = 1 / 200
int_time = 1e9  # in ns
collection = 0.01  # 1%


def calc_stochastic_matrix(green_rate, nir_rate):
    """Calculate the stochastic matrix for the NV center in a three-level model consisting
    of ground, excited, and singlet states. Time is discretized in 1 ns steps."""

    # Right stochastic matrix. Assume we are pumping into the excited singlet, which decays
    # immediately (~0.1 ns) into the ground singlet. Ground singlet then decays into ground triplet.
    return np.array(
        [
            [1 - (nir_rate + green_rate), green_rate, nir_rate],
            [exc_decay_rate + green_rate, 1 - (exc_decay_rate + green_rate), 0],
            [singlet_decay_rate, 0, 1 - singlet_decay_rate],
        ]
    )


def snr(green_rate, nir_rate):
    """Calculate the signal-to-noise ratio for a given green and NIR excitation rate.
    The amount of time spent in each state is calculated by finding the stationary distribution
    of the stochastic matrix that describes the dynamics of the system. Signal is the
    difference between the number of photons scattered with vs without NIR. Noise is
    shot noise in the reference case, without NIR.
    """

    # Signal experiment
    nir_on_matrix = calc_stochastic_matrix(green_rate, nir_rate)
    # Transpose the right stochastic matrix to a left stochastic matrix so that we can use eig
    nir_on_vals, nir_on_vecs = eig(nir_on_matrix.T)
    # Stationary distribution is the eigenvector with eigenvalue 1.
    nir_on_ind = np.where(np.isclose(nir_on_vals, 1))[0][0]
    nir_on_vec = nir_on_vecs.T[nir_on_ind]
    # Vector elements are probabilities so should sum to 1. Default norm is Euclidean distance=1
    nir_on_vec = nir_on_vec / np.sum(nir_on_vec)

    # Reference experiment
    nir_off_matrix = calc_stochastic_matrix(green_rate, 0)
    nir_off_vals, nir_off_vecs = eig(nir_off_matrix.T)
    nir_off_ind = np.where(np.isclose(nir_off_vals, 1))[0][0]
    nir_off_vec = nir_off_vecs.T[nir_off_ind]
    nir_off_vec = nir_off_vec / np.sum(nir_off_vec)

    # The total number of photons scattered is the total amount of time spent in
    # the excited state times the radiative decay rate out of the excited state.
    coeff = exc_decay_rate * int_time * collection
    signal = coeff * np.abs(nir_off_vec[1] - nir_on_vec[1])
    noise = np.sqrt(coeff * nir_on_vec[1])
    if signal == 0:
        ret_val = 0
    else:
        ret_val = signal / noise
    return ret_val


def main():
    """Plot SNR as a function of green and NIR excitation rates."""
    num_vals = 100
    snrs = np.empty((num_vals, num_vals))
    green_rate_vals = np.linspace(0.001, 0.1, num_vals)
    nir_rate_vals = np.linspace(0.0, 0.1e-5, num_vals)

    for green_ind in range(num_vals):
        for nir_ind in range(num_vals):
            green_rate = green_rate_vals[green_ind]
            nir_rate = nir_rate_vals[nir_ind]
            snr_val = snr(green_rate, nir_rate)
            snrs[green_ind, nir_ind] = snr_val

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(green_rate_vals * 1000, nir_rate_vals * 1e6, snrs.T)
    ax.set_xlabel("Green excitation rate (MHz)")
    ax.set_ylabel("NIR excitation rate (kHz)")
    fig.colorbar(mesh, label="SNR")

    fig, ax = plt.subplots()
    kpl.plot_line(ax, nir_rate_vals * 1e6, snrs[-1, :])
    ax.set_xlabel("NIR excitation rate (kHz)")
    ax.set_ylabel("SNR")
    kpl.anchored_text(ax, "100 MHz green excitation rate", kpl.Loc.UPPER_LEFT)


if __name__ == "__main__":
    kpl.init_kplotlib()
    main()
    snr()
    kpl.show(block=True)
