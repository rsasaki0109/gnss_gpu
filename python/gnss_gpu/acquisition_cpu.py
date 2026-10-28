"""One-period GPS C/A carrier-grid acquisition with whole-search calibration.

This extends the aperture delay fitter, not a tracking loop. Frequencies are
received carrier frequencies (IF plus Doppler), selected on a supplied grid.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from gnss_gpu.code_delay_cpu import (
    CodeDelayEstimate, _prepare_ca_observation, _estimate_prepared_delay,
)
from gnss_gpu.code_detection_cpu import (
    CaDetectionThreshold, ca_detection_statistic, calibrate_ca_threshold,
)
from gnss_gpu.input_validation import positive_float, positive_int
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index


@dataclass(frozen=True)
class CaSearch:
    sampling_freq: float
    carrier_grid_hz: tuple
    prn: int = 1

    def __post_init__(self):
        fs = positive_float("sampling_freq", self.sampling_freq)
        ratio = fs/CHIP_RATE
        if round(ratio) < 2 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-10):
            raise ValueError("sampling_freq must give >=2 integer samples/chip")
        prn = _sample_index("prn", self.prn)
        if not 1 <= prn <= 32:
            raise ValueError("prn must be an integer in 1..32")
        try:
            grid = np.asarray(self.carrier_grid_hz, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError("carrier_grid_hz must be a finite increasing 1D grid") from exc
        if (grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid))
                or np.any(np.diff(grid) <= 0)):
            raise ValueError("carrier_grid_hz must be a finite increasing 1D grid")
        if np.any(np.abs(grid) >= fs/2):
            raise ValueError("carrier_grid_hz must lie strictly inside the Nyquist interval")
        object.__setattr__(self, "sampling_freq", fs)
        object.__setattr__(self, "prn", prn)
        object.__setattr__(self, "carrier_grid_hz", tuple(float(x) for x in grid))


@dataclass(frozen=True)
class CaAcquisition:
    carrier_hz: float
    delay: CodeDelayEstimate
    statistic: float


def acquire_ca_signal(iq, search: CaSearch, *, workers=1):
    """Maximize fitted coherent energy over delay AND every carrier grid bin.

    Uses one 1 ms centered-aperture block, constant nav bit and negligible code
    Doppler during that block. No true delay/frequency is supplied. No carrier
    interpolation is performed. Off-grid carriers retain residual rotation.
    Samples must satisfy estimate_ca_delay's aperture and sampling assumptions.
    Equal scores select the lowest frequency. workers>1 threads independent
    bins; benefit depends on FFT size and thread/BLAS configuration. Numerical
    selection order is unchanged. All-zero IQ returns None; noise gives a
    candidate and requires an independently calibrated detection gate.
    """
    workers = positive_int("workers", workers)
    # One private snapshot prevents caller mutations between frequency bins.
    data = np.array(iq, copy=True)
    data.setflags(write=False)
    prepared = _prepare_ca_observation(data, search.sampling_freq, prn=search.prn)

    def fit(carrier):
        candidate = _estimate_prepared_delay(prepared, carrier_hz=carrier)
        if candidate is None:
            return None
        return CaAcquisition(carrier, candidate, ca_detection_statistic(candidate))

    if workers == 1:
        results = list(map(fit, search.carrier_grid_hz))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(search.carrier_grid_hz))) as pool:
            results = list(pool.map(fit, search.carrier_grid_hz))
    candidates = [result for result in results if result is not None]
    return max(candidates, key=lambda result: result.statistic) if candidates else None


@dataclass(frozen=True)
class CaSearchThreshold:
    search: CaSearch
    calibration: CaDetectionThreshold

    def __post_init__(self):
        if (self.calibration.sampling_freq != self.search.sampling_freq
                or self.calibration.prn != self.search.prn
                or self.calibration.carrier_hz != 0.):
            raise ValueError("calibration metadata must match search sampling/PRN and use carrier_hz=0")


def calibrate_ca_search(noise_statistics, search: CaSearch, *, false_alarm_probability=.01):
    """Freeze a rank threshold using max scores from the COMPLETE matching grid.

    Reusing single-frequency scores or changing the grid after calibration
    invalidates its false-alarm interpretation. Null blocks must be independent
    of evaluation and representative of the future noise distribution. The
    rank bound is marginal over calibration sets, not a per-threshold guarantee.
    """
    return CaSearchThreshold(search, calibrate_ca_threshold(
        noise_statistics, search.sampling_freq, prn=search.prn,
        false_alarm_probability=false_alarm_probability))


@dataclass(frozen=True)
class CaSearchDetection:
    accepted: bool
    statistic: float
    threshold: float
    candidate: Optional[CaAcquisition]


def detect_ca_search(iq, threshold: CaSearchThreshold, *, workers=1):
    """Run exactly the calibrated grid; preserve rejected candidates for audit."""
    candidate = acquire_ca_signal(iq, threshold.search, workers=workers)
    score = candidate.statistic if candidate is not None else 0.
    return CaSearchDetection(candidate is not None and score > threshold.calibration.score,
                             score, threshold.calibration.score, candidate)
