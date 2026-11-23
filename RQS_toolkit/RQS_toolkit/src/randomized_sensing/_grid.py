"""Bounded-memory multisine sampling and exact integrals of repaired clock cells."""

import numpy as np
from scipy.fft import irfft, rfft


# Roughly 64 MB of cached trigonometric tables, plus bounded FFT/GEMM batches.
_TABLE_ELEMENTS = 4_000_000
_BLOCK_ELEMENTS = 2_000_000


def fft_grid(frequencies, duration, length):
    """Return exact tone bins and transform length, or None for off-grid tones."""
    for factor in (1, 2, 4, 8, 16):
        cycles = factor*frequencies*duration
        bins = np.rint(cycles).astype(np.int64)
        tolerance = 32*np.finfo(float).eps*np.maximum(1., abs(cycles))
        if (np.all(abs(cycles-bins) <= tolerance) and np.all(bins > 0)
                and np.all(2*bins < factor*length) and len(np.unique(bins)) == len(bins)):
            n = factor*length
            divisor = int(np.gcd.reduce(np.r_[n, bins]))
            return bins//divisor, n//divisor
    return None


class CellGrid:
    def __init__(self, experiment, duration, start, tick, length):
        self.f = np.asarray(experiment.frequencies_hz)
        self.start, self.tick, self.length = start, tick, length
        self.fft = fft_grid(self.f, duration, length)
        self.cell = 2*np.pi*experiment.coupling_hz*tick*np.sinc(tick*self.f)
        self.phase = np.exp(2j*np.pi*self.f*(start+tick/2))
        self._cached = None
        if self.fft is None and len(self.f)*length <= _TABLE_ELEMENTS:
            self._cached = self._tables(0, length)

    @property
    def method(self):
        return "fft" if self.fft is not None else "dense"

    @property
    def batch_size(self):
        width = max(2*self.fft[1], 2*self.length) if self.fft is not None else self.length
        return max(1, min(256, _BLOCK_ELEMENTS//max(width, len(self.f))))

    def _tables(self, begin, end):
        t = self.start+(np.arange(begin, end)+.5)*self.tick
        angle = 2*np.pi*self.f[:, None]*t
        return np.cos(angle), np.sin(angle)

    def _blocks(self):
        if self._cached is not None:
            yield 0, self.length, self._cached
        else:
            width = max(1, _TABLE_ELEMENTS//len(self.f))
            for begin in range(0, self.length, width):
                end = min(self.length, begin+width)
                yield begin, end, self._tables(begin, end)

    def samples(self, theta):
        """Same iid phases as the dense multisine, sampled at cell midpoints."""
        if self.fft is not None:
            bins, n = self.fft
            spectrum = np.zeros((len(theta), n//2+1), complex)
            spectrum[:, bins] = np.exp(1j*theta)*self.phase
            values = irfft(spectrum, n=n, axis=1)*(n/2)
            if n < self.length:
                values = np.tile(values, (1, (self.length+n-1)//n))
            return values[:, :self.length]
        c, s = np.cos(theta), np.sin(theta)
        values = np.empty((len(theta), self.length))
        for begin, end, (ct, st) in self._blocks():
            values[:, begin:end] = c@ct-s@st
        return values

    def responses(self, patterns):
        """Integrate the signed, repaired +/-1 cells; no quadrature approximation."""
        if self.fft is not None:
            bins, n = self.fft
            if n < self.length:
                full, remainder = divmod(self.length, n)
                signed = patterns[:, :full*n].reshape(len(patterns), full, n).sum(axis=1, dtype=float)
                if remainder:
                    signed[:, :remainder] += patterns[:, full*n:]
            else:
                signed = patterns.astype(float)
            z = np.conj(rfft(signed, n=n, axis=1)[:, bins])
            z *= self.phase*self.cell
            return np.c_[z.real, z.imag]
        values = np.zeros((len(patterns), 2*len(self.f)))
        for begin, end, (ct, st) in self._blocks():
            signed = patterns[:, begin:end].astype(float)
            values[:, :len(self.f)] += signed@ct.T*self.cell
            values[:, len(self.f):] += signed@st.T*self.cell
        return values
