"""Known frequencies and hardware settings; observables are supplied after acquisition."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Experiment:
    """b(t)=sum(a*cos(2*pi*f*t)+b*sin(2*pi*f*t)); H=2*pi*coupling*b(t)*Z/2."""

    frequencies_hz: tuple[float, ...]
    coupling_hz: float
    t2_s: float | None = None
    amplitude_bound: float = 1.0
    min_gap_s: float = 20e-9
    edge_guard_s: float = 20e-9
    clock_s: float | None = None
    visibility: float = 1.0
    readout_bias: float = 0.0
    overhead_s: float = 0.0

    def __post_init__(self):
        f = np.asarray(self.frequencies_hz, float)
        if (f.ndim != 1 or not f.size or not np.isfinite(f).all()
                or np.any(f <= 0) or len(np.unique(f)) != len(f)):
            raise ValueError("frequencies_hz must be a nonempty vector of distinct positive frequencies.")
        object.__setattr__(self, "frequencies_hz", tuple(f.tolist()))
        for name in ("coupling_hz", "amplitude_bound"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("min_gap_s", "edge_guard_s", "overhead_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        for name in ("t2_s", "clock_s"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite, or None.")
        if not np.isfinite(self.visibility) or not 0 < self.visibility <= 1:
            raise ValueError("visibility must be in (0, 1].")
        if not np.isfinite(self.readout_bias) or abs(self.readout_bias)+self.visibility > 1:
            raise ValueError("Require |readout_bias| + visibility <= 1.")

    def contrast(self, duration_s):
        return self.visibility*np.exp(-duration_s/self.t2_s) if self.t2_s else self.visibility

    def to_dict(self):
        return asdict(self)
