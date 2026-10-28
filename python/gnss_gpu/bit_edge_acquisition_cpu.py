"""Coherent multi-period C/A acquisition with an unknown single bit reversal.

Search one omitted transition period plus the constant-polarity hypothesis.
All carrier, delay and edge trials belong to one calibrated decision.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.code_delay_cpu import _cached_reference_data, _build_reference_data, _CACHE_MAX_SAMPLES
from gnss_gpu.input_validation import positive_int
from gnss_gpu.input_validation import positive_float
from gnss_gpu.noncoherent_tolerance_cpu import tolerance_order
from gnss_gpu.signal_sim_cpu import CHIP_RATE


@dataclass(frozen=True)
class CaBitEdgeSearch:
    search: CaSearch
    periods: int = 20
    residual_grid_hz: tuple = tuple(float(x) for x in range(-125, 126, 25))
    shortlist_size: Optional[int] = 64

    def __post_init__(self):
        periods = positive_int("periods", self.periods)
        offsets = np.asarray(self.residual_grid_hz, dtype=float)
        if periods < 3:
            raise ValueError("at least three periods required")
        if offsets.ndim != 1 or not len(offsets) or not np.all(np.isfinite(offsets)) or np.any(np.diff(offsets) <= 0):
            raise ValueError("residual_grid_hz must be finite, increasing and one-dimensional")
        if 0. not in offsets or np.any(abs(offsets) >= 500.):
            raise ValueError("residual grid must contain zero and stay inside +/-500 Hz")
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "residual_grid_hz", tuple(float(value) for value in offsets))
        if self.shortlist_size is not None:
            object.__setattr__(self, "shortlist_size", positive_int("shortlist_size", self.shortlist_size))


@dataclass(frozen=True)
class CaBitEdgeCandidate:
    carrier_hz: float
    delay_samples: int
    excluded_period: Optional[int]
    coarse_carrier_hz: float
    statistic: float


def acquire_ca_bit_edge(iq, config: CaBitEdgeSearch):
    """Coherently sum period correlations after slow-time carrier correction.

    Input is contiguous interleaved-real 1 ms rows. For every carrier/delay,
    consider no bit reversal or a reversal inside any one period. The latter
    period is omitted; rows on opposite sides are given opposite signs. Overall
    sign and carrier phase remain unknown. No bit timing or true carrier enters.

    At each coarse carrier, retain the configured strongest noncoherent delay
    bins before coherent search. Set shortlist_size=None for all delays. Pruning
    can miss a coherent peak; its data selection belongs in calibration too.
    Coarse carrier wiping acts within each period; residual-grid correction
    acts between periods. Offsets can therefore leave within-period rotation
    loss. Aperture integer-delay templates match the previous detector model.
    No code drift, multiple bit reversals, multipath or acceleration are fitted.
    """
    search = config.search
    spc = round(search.sampling_freq/CHIP_RATE)
    count = 1023*spc
    raw = np.asarray(iq)
    if raw.shape != (config.periods, 2*count) or np.iscomplexobj(raw):
        raise ValueError("iq must contain exactly the configured real interleaved 1 ms rows")
    data = np.array(raw, dtype=float, copy=True).reshape(config.periods, count, 2)
    if not np.all(np.isfinite(data)):
        raise ValueError("iq must be finite")
    scale = float(np.max(abs(data)))
    if not scale:
        return None
    signal = data[:, :, 0]/scale+1j*(data[:, :, 1]/scale)
    row_energy = np.sum(signal.real**2+signal.imag**2, axis=1)
    total_energy = float(np.sum(row_energy))
    samples, spectrum = (_cached_reference_data if count <= _CACHE_MAX_SAMPLES else _build_reference_data)(search.prn, spc)
    reference_energy = float(np.sum(abs(spectrum)**2)/count)
    edge_energy = total_energy-row_energy
    times = np.arange(config.periods)*.001
    best = None
    for coarse in search.carrier_grid_hz:
        wiped = signal*np.exp(-2j*np.pi*coarse*samples/search.sampling_freq)
        correlations = np.fft.ifft(np.fft.fft(wiped, axis=1)*spectrum, axis=1)
        lags = np.arange(count)
        if config.shortlist_size is not None and config.shortlist_size < count:
            power = np.sum(abs(correlations)**2, axis=0)
            lags = np.sort(np.argsort(-power, kind="stable")[:config.shortlist_size])
            correlations = correlations[:, lags]
        for offset in config.residual_grid_hz:
            carrier = coarse+offset
            if not search.carrier_grid_hz[0] <= carrier <= search.carrier_grid_hz[-1]:
                continue
            corrected = correlations*np.exp(-2j*np.pi*carrier*times[:, None])
            prefix = np.cumsum(corrected, axis=0)
            total = prefix[-1]
            sums = 2*prefix-corrected-total[None, :]
            edge_scores = np.divide(count*abs(sums)**2, reference_energy*edge_energy[:, None],
                                    out=np.zeros_like(sums.real), where=edge_energy[:, None] > 0)
            flat = int(np.argmax(edge_scores))
            edge, lag = np.unravel_index(flat, edge_scores.shape)
            score, omitted = float(edge_scores[edge, lag]), int(edge)
            constant_scores = count*abs(total)**2/(reference_energy*total_energy)
            constant_lag = int(np.argmax(constant_scores))
            if float(constant_scores[constant_lag]) >= score:
                score, lag, omitted = float(constant_scores[constant_lag]), constant_lag, None
            if best is None or score > best.statistic:
                selected_lag = int(lags[lag])
                best = CaBitEdgeCandidate(float(carrier), int((selected_lag+count//2) % count-count//2), omitted, coarse, score)
    return best


@dataclass(frozen=True)
class CaBitEdgeStratumCalibration:
    name: str
    score: float
    trials: int
    order_statistic: int
    failure_probability: float
    binomial_failure_bound: float


@dataclass(frozen=True)
class CaBitEdgeThreshold:
    config: CaBitEdgeSearch
    score: float
    strata: tuple
    max_false_alarm_probability: float
    joint_failure_budget: float
    methods_in_family: int


def calibrate_ca_bit_edge(scores_by_stratum, config, *, max_false_alarm_probability=.01,
                          joint_failure_budget=.05, methods_in_family=2):
    """Tolerance-calibrate all data-dependent search steps in each null stratum.

    Use one maximum gate, without a runtime stratum label. Family allocation
    includes every compared method and every declared null stratum.
    """
    methods = positive_int("methods_in_family", methods_in_family)
    budget = positive_float("joint_failure_budget", joint_failure_budget)
    if not scores_by_stratum or budget >= 1:
        raise ValueError("require nonempty strata and failure budget below one")
    delta = budget/(methods*len(scores_by_stratum))
    calibrated = []
    for name in sorted(scores_by_stratum):
        values = np.asarray(list(scores_by_stratum[name]), dtype=float)
        if not isinstance(name, str) or not name or values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("require named finite positive null scores")
        order, bound = tolerance_order(len(values), max_false_alarm_probability, delta)
        calibrated.append(CaBitEdgeStratumCalibration(name, float(np.partition(values,order-1)[order-1]), len(values), order, delta, bound))
    return CaBitEdgeThreshold(config, max(value.score for value in calibrated), tuple(calibrated),
                              float(max_false_alarm_probability), budget, methods)


@dataclass(frozen=True)
class CaBitEdgeDetection:
    accepted: bool
    statistic: float
    threshold: float
    candidate: Optional[CaBitEdgeCandidate]


def detect_ca_bit_edge(iq, threshold):
    candidate = acquire_ca_bit_edge(iq,threshold.config)
    score = candidate.statistic if candidate is not None else 0.
    return CaBitEdgeDetection(candidate is not None and score > threshold.score, score, threshold.score, candidate)
