"""Post-detection fractional delay using the acquired carrier/bit-edge hypothesis."""

import numpy as np

from gnss_gpu.code_delay_cpu import estimate_ca_delay


def refine_weak_ca_delay(iq, config, candidate):
    """Coherently combine aperture-sampled rows and fit a fractional delay.

    Call after the original calibrated acquisition gate accepts. This estimator
    supplies no new detection score or uncertainty. Candidate and IQ must come
    from the same search. Rectangular instantaneous samples cannot identify
    arbitrary fractional delays; centered integrate-and-dump input is required.
    Constant carrier/delay and at most one bit reversal are assumed.
    """
    fs = config.search.sampling_freq
    n = round(fs*.001)
    data = np.asarray(iq)
    if data.shape != (config.periods, 2*n) or np.iscomplexobj(data) or not np.all(np.isfinite(data)):
        raise ValueError('require finite real interleaved IQ matching the search')
    if candidate is None or not np.isfinite(candidate.carrier_hz):
        raise ValueError('require an acquired carrier/edge candidate')
    edge = candidate.excluded_period
    if edge is not None and (isinstance(edge, bool) or int(edge) != edge or not 0 <= edge < config.periods):
        raise ValueError('invalid excluded period')
    scale = float(np.max(abs(data)))
    if scale == 0:
        return None
    pairs = np.asarray(data, dtype=float).reshape(config.periods, n, 2)/scale
    signal = pairs[:,:,0]+1j*pairs[:,:,1]
    time = np.arange(config.periods*n).reshape(config.periods,n)/fs
    signal *= np.exp(-2j*np.pi*candidate.carrier_hz*time)
    signs = np.ones(config.periods)
    if edge is not None:
        signs[edge] = 0.
        signs[edge+1:] = -1.
    combined = np.sum(signal*signs[:,None], axis=0)
    packed = np.column_stack((combined.real,combined.imag)).ravel()
    return estimate_ca_delay(packed,fs,prn=config.search.prn)
