"""Consistent carrier and code phase for a constant excess path-length rate.

For L(t)=L0+v*t, relative carrier Doppler is -f_L1*v/c and the
baseband harmonic f additionally rotates by -f*v*t/c. The initial path
phase/delay remain the caller's responsibility. Common oscillator frequency
and linear drift are independent nuisance terms, not receiver motion.
This keeps the existing harmonic-observation approximation (no bin leakage).
"""
import numpy as np
from gnss_gpu.cached_drift_carrier_paths_cpu import _quadrature
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED

L1_FREQUENCY_HZ=1575.42e6


def range_rate_from_doppler(doppler_hz):
    if not np.isfinite(doppler_hz):raise ValueError('finite relative Doppler required')
    return -float(doppler_hz)*LIGHT_SPEED/L1_FREQUENCY_HZ


def moving_echo_blocks(frequencies_hz,common_hz,drift_hz_s,excess_range_rate_m_s,count=40):
    f=np.asarray(frequencies_hz,dtype=float)
    if f.ndim!=1 or not len(f) or not np.all(np.isfinite(f)):
        raise ValueError('finite nonempty harmonic frequency vector required')
    if not np.all(np.isfinite([common_hz,drift_hz_s,excess_range_rate_m_s])) or count not in (10,20,40):
        raise ValueError('finite motion/oscillator parameters and10/20/40blocks required')
    effective=common_hz-(L1_FREQUENCY_HZ+f)*excess_range_rate_m_s/LIGHT_SPEED
    t,w=_quadrature(count)
    result=np.empty((count,len(f)),dtype=complex)
    # Bound temporary quadrature memory rather than allocate all harmonics at once.
    for start in range(0,len(f),512):
        frequency=effective[start:start+512]
        phase=2j*np.pi*(t[:,:,None]*frequency+.5*drift_hz_s*t[:,:,None]**2)
        result[:,start:start+512]=np.einsum('bqf,q->bf',np.exp(phase),w)
    return result
