"""Controlled passband noise shapes with equal mean spectral variance.

These are synthetic stress profiles, not measured RF/frontend/weather models.
Independent spectral coefficients with nonuniform variance imply colored noise
in time. Colored profiles have no single frequency-independent noise PSD.
"""

import numpy as np


def spectral_noise_shape(frequencies_hz, profile="white"):
    """Return nonnegative variance weights normalized to arithmetic mean one.

    Profiles use the largest represented |frequency| as their scale. The narrow
    center bump has width cutoff/12. Equal mean weight fixes expected total
    passband noise energy, not delay-estimation information.
    """
    frequency = np.asarray(frequencies_hz, dtype=float)
    if frequency.ndim != 1 or len(frequency) < 2 or not np.all(np.isfinite(frequency)):
        raise ValueError("finite frequency vector of length >=2 required")
    cutoff = float(np.max(np.abs(frequency)))
    if cutoff <= 0:
        raise ValueError("nonzero frequency span required")
    x = frequency / cutoff
    if profile == "white":
        weights = np.ones_like(x)
    elif profile == "center":
        weights = 1 + 9 * np.exp(-0.5 * (12 * x) ** 2)
    elif profile == "edge":
        weights = 1 + 9 * x**4
    else:
        raise ValueError("profile must be white, center or edge")
    return weights / weights.mean()
