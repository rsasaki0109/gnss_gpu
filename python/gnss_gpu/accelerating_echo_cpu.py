"""Quadratic excess range, with consistent carrier and code phase.

Initial delay and phase are supplied by the caller. This is a harmonic
observation model, not an RF resampler or an inter-bin leakage model.
"""
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import _quadrature
from gnss_gpu.moving_echo_cpu import L1_FREQUENCY_HZ, moving_echo_blocks
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED


def minimum_excess_range(initial_m, velocity_m_s, acceleration_m_s2, duration_s):
    """Minimum L0 + v*t + a*t**2/2 over the entire closed interval."""
    if not np.all(np.isfinite([initial_m, velocity_m_s, acceleration_m_s2, duration_s])) or duration_s < 0:
        raise ValueError('finite range/motion and nonnegative duration required')
    times = [0., duration_s]
    if acceleration_m_s2 > 0:
        vertex = -velocity_m_s / acceleration_m_s2
        if 0 < vertex < duration_s:
            times.append(vertex)
    return min(initial_m + velocity_m_s*t + .5*acceleration_m_s2*t*t for t in times)


def accelerating_echo_blocks(frequencies_hz, common_hz, drift_hz_s,
                             excess_range_rate_m_s, excess_acceleration_m_s2, count=40):
    f = np.asarray(frequencies_hz, dtype=float)
    if f.ndim != 1 or not len(f) or not np.all(np.isfinite(f)):
        raise ValueError('finite nonempty harmonic frequency vector required')
    if (not np.all(np.isfinite([common_hz, drift_hz_s, excess_range_rate_m_s, excess_acceleration_m_s2]))
            or not isinstance(count, (int, np.integer)) or isinstance(count, (bool, np.bool_))
            or count not in (10, 20, 40)):
        raise ValueError('finite motion/oscillator parameters and integer 10/20/40 blocks required')
    if excess_acceleration_m_s2 == 0:
        return moving_echo_blocks(f, common_hz, drift_hz_s, excess_range_rate_m_s, count)
    effective_frequency = common_hz - (L1_FREQUENCY_HZ+f)*excess_range_rate_m_s/LIGHT_SPEED
    effective_drift = drift_hz_s - (L1_FREQUENCY_HZ+f)*excess_acceleration_m_s2/LIGHT_SPEED
    t, w = _quadrature(count)
    result = np.empty((count, len(f)), dtype=complex)
    for start in range(0, len(f), 512):
        phase = 2j*np.pi*(t[:, :, None]*effective_frequency[start:start+512]
                         + .5*t[:, :, None]**2*effective_drift[start:start+512])
        result[:, start:start+512] = np.einsum('bqf,q->bf', np.exp(phase), w)
    return result
