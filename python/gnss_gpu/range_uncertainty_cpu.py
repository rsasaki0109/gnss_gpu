"""Causal range variability estimate for the configured CPU smoothing model.

This estimates short-term scatter, not multipath bias or acquisition ambiguity.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from gnss_gpu.input_validation import finite_float, positive_float, positive_int


@dataclass(frozen=True)
class RangeUncertaintyConfig:
    gain: float = .25
    window_differences: int = 32
    min_differences: int = 8
    initial_sigma_m: float = 10.
    min_sigma_m: float = 2.
    max_sigma_m: float = 100.
    max_gap_s: float = .005

    def __post_init__(self):
        for name in ("gain", "initial_sigma_m", "min_sigma_m", "max_sigma_m", "max_gap_s"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        for name in ("window_differences", "min_differences"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))
        if (self.gain > 1 or self.min_differences < 2 or self.window_differences < self.min_differences
                or not self.min_sigma_m <= self.initial_sigma_m <= self.max_sigma_m):
            raise ValueError("invalid smoothing gain, window or sigma limits")


@dataclass(frozen=True)
class RangeUncertainty:
    sigma_m: float
    prior_differences: int
    ready: bool


class CausalRangeUncertainty:
    """Per-PRN prior-only MAD of consecutive smoothed range differences.

    For stationary x_t=(1-a)x_(t-1)+a*z_t with independent white z_t,
    Var(x_t-x_(t-1))=2*a*Var(x_t). Thus sigma_x is Gaussian-consistent
    MAD(delta_x)/sqrt(2*a). Trend, path bias, shared errors and reacquisition
    invalidate that interpretation. No detection statistic is treated as C/N0.
    The current range updates history only AFTER its sigma is computed. A long
    gap restarts the cold prior. Call only for fresh measurements.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else RangeUncertaintyConfig()
        if not isinstance(self.config, RangeUncertaintyConfig):
            raise TypeError("config must be RangeUncertaintyConfig")
        self.reset()

    def reset(self):
        self._channels = {}

    def update(self, prn, pseudorange_m, *, time_s):
        key = positive_int("prn", prn)
        value, time = positive_float("pseudorange_m", pseudorange_m), finite_float("time_s", time_s)
        previous = self._channels.get(key)
        cfg = self.config
        if previous is not None and time <= previous[0]:
            raise ValueError("time_s must strictly increase per PRN")
        if previous is None or time-previous[0] > cfg.max_gap_s:
            differences = deque(maxlen=cfg.window_differences)
            previous = None
        else:
            differences = previous[2]
        count = len(differences)
        sigma = cfg.initial_sigma_m
        if count >= cfg.min_differences:
            values = np.asarray(differences)
            sigma = float(1.482602218505602*np.median(np.abs(values-np.median(values)))/np.sqrt(2*cfg.gain))
            sigma = float(np.clip(sigma, cfg.min_sigma_m, cfg.max_sigma_m))
        result = RangeUncertainty(sigma, count, count >= cfg.min_differences)
        if previous is not None:
            differences.append(value-previous[1])
        self._channels[key] = (time, value, differences)
        return result
