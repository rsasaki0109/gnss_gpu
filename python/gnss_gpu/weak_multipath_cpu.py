"""Post-acquisition two-path diagnostics on coherent aperture observations.

Original acquisition acceptance is unchanged. LOS-dominant local fitting can
reject ambiguous delays and cannot guarantee separation or multipath detection.
"""
import numpy as np
from gnss_gpu.multipath_fit_cpu import fit_ca_two_paths


def coherent_weak_period(iq, config, candidate):
    """Combine rows using only the acquired carrier and bit-edge hypothesis."""
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
    return packed


def fit_weak_ca_two_paths(iq, config, candidate, **kwargs):
    period = coherent_weak_period(iq, config, candidate)
    if period is None:
        return None
    return fit_ca_two_paths(period,config.search.sampling_freq,prn=config.search.prn,**kwargs)
