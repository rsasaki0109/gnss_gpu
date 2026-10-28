"""Fixed-window noncoherent C/A acquisition on integer circular delay bins.

Accumulate correlation powers at the SAME delay and carrier before maximizing.
Independent per-period peak selection is not noncoherent signal integration.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.code_delay_cpu import _cached_reference_data, _build_reference_data, _CACHE_MAX_SAMPLES
from gnss_gpu.code_detection_cpu import CaDetectionThreshold, calibrate_ca_threshold
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE


@dataclass(frozen=True)
class CaNoncoherentSearch:
    search: CaSearch
    periods_per_window: int = 4
    windows_per_decision: int = 1

    def __post_init__(self):
        for name in ("periods_per_window", "windows_per_decision"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))


@dataclass(frozen=True)
class CaNoncoherentCandidate:
    carrier_hz: float
    delay_samples: int
    window_index: int
    statistic: float


def acquire_ca_noncoherent(iq, config: CaNoncoherentSearch):
    """Search all configured windows/carriers/integer delays in one decision.

    IQ has shape (periods_per_window*windows_per_decision, 2*samples_per_ms),
    with real interleaved I/Q per row, chronological contiguous 1 ms periods.
    Timestamps/continuity are the caller's responsibility. Templates use the
    same centered aperture as the existing fitter, but delays are integer bins.

    Each window uses one common code delay and carrier. No code drift, carrier
    acceleration or fractional-delay fitting is applied. Whole-period phase or
    nav-polarity changes are allowed; a nav transition WITHIN a period can still
    lose signal power. The output is the strongest candidate, not a detection.
    Ties select the lowest carrier, first window, then first circular lag bin.
    """
    search = config.search
    count = 1023*round(search.sampling_freq/CHIP_RATE)
    periods = config.periods_per_window*config.windows_per_decision
    raw = np.asarray(iq)
    if raw.shape != (periods, 2*count) or np.iscomplexobj(raw):
        raise ValueError("iq must contain the configured number of real interleaved 1 ms rows")
    data = np.array(raw, dtype=np.float64, copy=True).reshape(periods, count, 2)
    if not np.all(np.isfinite(data)):
        raise ValueError("iq must be finite")
    # Normalize components before building complex values or inner products,
    # including finite inputs close to float64's largest representable value.
    scale = float(np.max(np.abs(data)))
    if scale == 0:
        return None
    signal = data[:, :, 0]/scale+1j*(data[:, :, 1]/scale)
    k, windows = config.periods_per_window, config.windows_per_decision
    energy = np.sum(signal.real**2+signal.imag**2, axis=1).reshape(windows, k).sum(axis=1)
    samples, spectrum = (_cached_reference_data if count <= _CACHE_MAX_SAMPLES else _build_reference_data)(
        search.prn, round(search.sampling_freq/CHIP_RATE))
    reference_energy = float(np.sum(spectrum.real**2+spectrum.imag**2)/count)
    best = None
    for carrier in search.carrier_grid_hz:
        wiped = signal*np.exp(-2j*np.pi*carrier*samples/search.sampling_freq)
        correlations = np.fft.ifft(np.fft.fft(wiped, axis=1)*spectrum, axis=1)
        power = (correlations.real**2+correlations.imag**2).reshape(windows, k, count).sum(axis=1)
        scores = np.divide(count*k*power, reference_energy*energy[:, None],
                           out=np.zeros_like(power), where=energy[:, None] > 0)
        window, lag = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[window, lag])
        if best is None or score > best.statistic:
            best = CaNoncoherentCandidate(carrier, int((lag+count//2) % count-count//2), int(window), score)
    return best


@dataclass(frozen=True)
class CaNoncoherentThreshold:
    config: CaNoncoherentSearch
    calibration: CaDetectionThreshold

    def __post_init__(self):
        search = self.config.search
        if (self.calibration.sampling_freq != search.sampling_freq or self.calibration.prn != search.prn
                or self.calibration.carrier_hz != 0.):
            raise ValueError("calibration metadata must match the configured search")


def calibrate_ca_noncoherent(noise_statistics, config, *, false_alarm_probability=.01):
    """Calibrate maxima over the COMPLETE configured decision, including windows.

    Supply independent noise-only decision scores from this exact configuration.
    Changing period count, grid, window count or scoring needs new calibration.
    The rank bound is marginal over calibration sets, not an integrity guarantee.
    The reused rank metadata's sampling frequency describes each 1 ms row;
    config binds the full integration duration and number of searched windows.
    """
    return CaNoncoherentThreshold(config, calibrate_ca_threshold(noise_statistics,
        config.search.sampling_freq, prn=config.search.prn, false_alarm_probability=false_alarm_probability))


@dataclass(frozen=True)
class CaNoncoherentDetection:
    accepted: bool
    statistic: float
    threshold: float
    candidate: Optional[CaNoncoherentCandidate]


def detect_ca_noncoherent(iq, threshold: CaNoncoherentThreshold):
    candidate = acquire_ca_noncoherent(iq, threshold.config)
    score = candidate.statistic if candidate is not None else 0.
    return CaNoncoherentDetection(candidate is not None and score > threshold.calibration.score,
                                  score, threshold.calibration.score, candidate)
