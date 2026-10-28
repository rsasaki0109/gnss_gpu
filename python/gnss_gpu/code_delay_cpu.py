"""Fractional GPS C/A delay fitting for integrate-and-dump sampled IQ.

The acquisition model is one constant complex-amplitude path, one code period,
integer samples/chip and a centered sampling aperture of one sample interval.
It uses no propagation geometry or true delay. Multipath remains a model error.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from gnss_gpu.input_validation import finite_float, positive_float
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index
from gnss_gpu.validation.tracking_residuals import generate_ca_code

C_LIGHT = 299792458.0


@dataclass(frozen=True)
class CodeDelayEstimate:
    delay_samples: float
    sampling_freq: float
    coarse_delay_samples: int
    amplitude: complex
    fit_residual_fraction: float

    @property
    def delay_m(self):
        return self.delay_samples*C_LIGHT/self.sampling_freq


def _averaged_template(code, samples_per_chip, delay_samples):
    phase = (np.arange(1023*samples_per_chip)-delay_samples)/samples_per_chip
    left = phase-.5/samples_per_chip
    first = np.floor(left).astype(np.int64)
    weight = np.clip((first+1-left)*samples_per_chip, 0., 1.)
    return weight*code[first % 1023]+(1-weight)*code[(first+1) % 1023]


# Limit retained arrays by both entry count and samples/code period. Oversized
# periods use the original uncached construction rather than growing memory.
_CACHE_MAX_SAMPLES = 8192


def _build_reference_data(prn, samples_per_chip):
    samples = np.arange(1023*samples_per_chip)
    local = _averaged_template(generate_ca_code(prn), samples_per_chip, 0.)
    spectrum = np.conj(np.fft.fft(local))
    samples.setflags(write=False)
    spectrum.setflags(write=False)
    return samples, spectrum


def _build_interval_data(prn, samples_per_chip, center):
    code = generate_ca_code(prn)
    left = center-.5
    a = _averaged_template(code, samples_per_chip, left)
    b = _averaged_template(code, samples_per_chip, left+1)-a
    aa, ab, bb = float(a@a), float(a@b), float(b@b)
    a.setflags(write=False)
    b.setflags(write=False)
    return a, b, aa, ab, bb


_cached_reference_data = lru_cache(maxsize=16)(_build_reference_data)
_cached_interval_data = lru_cache(maxsize=64)(_build_interval_data)


def clear_ca_delay_cache():
    """Release retained reference/interval arrays; ongoing fits retain their views.

    No input IQ, carrier estimates, fitted projections or decisions are cached.
    References are immutable. LRU caches are safe for concurrent acquisitions.
    """
    _cached_reference_data.cache_clear()
    _cached_interval_data.cache_clear()


def estimate_ca_delay(iq, sampling_freq, *, prn=1, carrier_hz=0.):
    """Fit fractional code delay after coarse circular FFT acquisition.

    IQ must be one interleaved I/Q period (1 ms), centered integrate-and-dump
    samples, >=2 integer samples/chip. carrier_hz wipes a supplied residual
    carrier; no carrier search or code-Doppler correction is performed. Template
    fitting neglects carrier rotation within each aperture, so the carrier
    frequency at sampling must be small compared with sampling_freq.
    The code delay is wrapped into +/-0.5 ms. Unknown constant phase, amplitude
    and nav-bit polarity are fitted jointly as a complex amplitude.

    Within each half-integer-bounded sample-delay interval the aperture-averaged
    template is affine. Maximize normalized coherent power analytically in the
    three intervals around the strongest FFT bin. This refines that peak; it
    does not resolve multiple paths or prove signal detection. All-zero input
    returns None. fit_residual_fraction measures waveform mismatch, not ranging
    uncertainty or detection confidence.
    """
    prepared = _prepare_ca_observation(iq, sampling_freq, prn=prn)
    return _estimate_prepared_delay(prepared, carrier_hz=carrier_hz)


@dataclass(frozen=True)
class _CaObservation:
    sampling_freq: float
    samples_per_chip: int
    prn: int
    signal: np.ndarray
    scale: float


def _prepare_ca_observation(iq, sampling_freq, *, prn=1):
    """Own a normalized, immutable observation for one local frequency search.

    This object is never retained in the global reference caches. Each trial
    creates its own carrier-wiped signal, including in parallel searches.
    """
    fs = positive_float("sampling_freq", sampling_freq)
    ratio = fs/CHIP_RATE
    samples_per_chip = round(ratio)
    if samples_per_chip < 2 or not math.isclose(ratio, samples_per_chip, rel_tol=0, abs_tol=1e-10):
        raise ValueError("sampling_freq must give >=2 integer samples/chip")
    prn = _sample_index("prn", prn)
    if not 1 <= prn <= 32:
        raise ValueError("prn must be an integer in 1..32")
    data = np.asarray(iq)
    count = 1023*samples_per_chip
    if data.ndim != 1 or data.size != 2*count or np.iscomplexobj(data):
        raise ValueError("iq must be one 1 ms period of interleaved real I/Q")
    data = np.asarray(data, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(data)):
        raise ValueError("iq must be finite")
    signal = data[:, 0]+1j*data[:, 1]
    scale = float(np.max(np.abs(signal)))
    # Scale before inner products to keep finite large/small IQ well conditioned.
    if scale:
        signal = signal/scale
    signal.setflags(write=False)
    return _CaObservation(fs, samples_per_chip, prn, signal, scale)


def _estimate_prepared_delay(prepared, *, carrier_hz=0.):
    fs, samples_per_chip, prn = prepared.sampling_freq, prepared.samples_per_chip, prepared.prn
    carrier = finite_float("carrier_hz", carrier_hz)
    if abs(carrier) > fs/2:
        raise ValueError("carrier_hz must lie within the Nyquist interval")
    scale = prepared.scale
    if scale == 0:
        return None
    count = 1023*samples_per_chip
    use_cache = count <= _CACHE_MAX_SAMPLES
    samples, reference_spectrum = (_cached_reference_data if use_cache else _build_reference_data)(prn, samples_per_chip)
    signal = prepared.signal*np.exp(-2j*np.pi*carrier*samples/fs)
    energy = float(np.vdot(signal, signal).real)
    corr = np.fft.ifft(np.fft.fft(signal)*reference_spectrum)
    coarse = int(np.argmax(np.abs(corr)))
    best_score, best_delay, best_amplitude = -1., float(coarse), 0j
    for center in range(coarse-1, coarse+2):
        left = center-.5
        a, b, aa, ab, bb = (_cached_interval_data if use_cache else _build_interval_data)(prn, samples_per_chip, center)
        p, q = np.dot(a, signal), np.dot(b, signal)
        n0, n1, n2 = abs(p)**2, 2*float((np.conj(p)*q).real), abs(q)**2
        # Stationary points of |p+w*q|^2 / ||a+w*b||^2.
        polynomial = np.array([2*n2*ab-n1*bb, 2*(n2*aa-n0*bb), n1*aa-2*n0*ab])
        candidates = [0., 1.]
        norm = float(np.max(np.abs(polynomial)))
        if norm:
            # np.roots already strips leading zeros, including a degenerate
            # quadratic's zero leading coefficient. Keep its solver and order.
            roots = np.roots(polynomial/norm)
            candidates.extend(float(root.real) for root in roots
                              if abs(root.imag) < 1e-10 and 0 < root.real < 1)
        for weight in candidates:
            template_energy = aa+2*ab*weight+bb*weight*weight
            projection = p+weight*q
            score = float(abs(projection)**2/template_energy)
            if score > best_score:
                best_score = score
                best_delay = left+weight
                best_amplitude = complex(projection/template_energy)*scale
    wrapped = (best_delay+count/2) % count-count/2
    return CodeDelayEstimate(float(wrapped), fs,
                             int((coarse+count//2) % count-count//2), best_amplitude,
                             float(np.clip(1-best_score/energy, 0., 1.)))
