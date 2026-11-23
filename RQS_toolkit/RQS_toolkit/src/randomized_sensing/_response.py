"""Closed-form response of +/-1 intervals: the independent verifier for pool entries."""

import numpy as np


def _integrals(experiment, left, right, start=0.0, dtype=float):
    """Exact Fourier integrals over [left, right) intervals; shape (intervals, 2N)."""
    f = np.asarray(experiment.frequencies_hz, dtype=dtype)
    left, right = np.atleast_1d(left).astype(dtype), np.atleast_1d(right).astype(dtype)
    width = right - left
    pi = np.arccos(dtype(-1))
    angle = 2*pi*(dtype(start) + (left + right)/2)[:, None]*f
    scale = width[:, None]*np.sinc(width[:, None]*f)
    return np.c_[scale*np.cos(angle), scale*np.sin(angle)]


def response(experiment, switch_times_s, duration_s, start_time_s=0.0):
    """Reintegrate actual +/-1 intervals, using extended precision where available."""
    edges = np.r_[0.0, switch_times_s, duration_s]
    integrals = _integrals(experiment, edges[:-1], edges[1:], start_time_s, np.longdouble)
    pi = np.arccos(np.longdouble(-1))
    return np.asarray(2*pi*experiment.coupling_hz*((-1.0)**np.arange(len(integrals)) @ integrals), float)
