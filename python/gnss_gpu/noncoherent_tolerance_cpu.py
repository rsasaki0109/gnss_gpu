"""Distribution-free upper-tail tolerance calibration for a fixed search.

Under IID representative null records, the conditional false-alarm probability
of the frozen threshold exceeds alpha with calibration probability <= delta.
This differs from the existing marginal rank calibration. No SciPy is needed.
"""

from dataclasses import dataclass
import math

import numpy as np

from gnss_gpu.input_validation import positive_float, positive_int
from gnss_gpu.noncoherent_acquisition_cpu import (
    CaNoncoherentSearch, CaNoncoherentDetection, acquire_ca_noncoherent,
)


def binomial_lower_tail(n, probability, upper):
    """P[Binomial(n, probability) <= upper], with log-domain summation."""
    if upper < 0:
        return 0.
    if upper >= n:
        return 1.
    logs = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
            +j*math.log(probability)+(n-j)*math.log1p(-probability)
            for j in range(upper+1)]
    largest = max(logs)
    return min(1., math.exp(largest)*math.fsum(math.exp(value-largest) for value in logs))


def tolerance_order(trials, max_false_alarm_probability=.01, failure_probability=.025):
    """Smallest ascending rank k with P[Bin(n, alpha) <= n-k] <= delta.

    For continuous null scores this is the probability that the kth score
    undershoots the (1-alpha) quantile. Strict > acceptance makes ties conservative.
    """
    n = positive_int("trials", trials)
    alpha = positive_float("max_false_alarm_probability", max_false_alarm_probability)
    delta = positive_float("failure_probability", failure_probability)
    if alpha >= 1 or delta >= 1:
        raise ValueError("probabilities must be below one")
    if binomial_lower_tail(n, alpha, 0) > delta:
        raise ValueError("insufficient calibration records for requested tolerance")
    left, right = 1, n
    while left < right:
        middle = (left+right)//2
        if binomial_lower_tail(n, alpha, n-middle) <= delta:
            right = middle
        else:
            left = middle+1
    return left, binomial_lower_tail(n, alpha, n-left)


@dataclass(frozen=True)
class CaNoncoherentToleranceThreshold:
    config: CaNoncoherentSearch
    score: float
    calibration_trials: int
    order_statistic: int
    max_false_alarm_probability: float
    calibration_failure_probability: float
    binomial_failure_bound: float


def calibrate_ca_noncoherent_tolerance(noise_statistics, config, *,
                                       max_false_alarm_probability=.01, failure_probability=.025):
    """Freeze a tolerance threshold using full-decision null maxima only.

    Scores must come from independent records evaluated with this exact config.
    The confidence statement assumes the calibration and future null records
    share a distribution. It does not cover interference, RF mismatch or tracks.
    With two methods, delta=.025 each gives a joint failure bound of .05 by
    the union bound, even when the calibration records are shared.
    """
    values = np.asarray(list(noise_statistics), dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("noise_statistics must contain finite positive scores")
    order, bound = tolerance_order(len(values), max_false_alarm_probability, failure_probability)
    return CaNoncoherentToleranceThreshold(config, float(np.partition(values, order-1)[order-1]),
        len(values), order, float(max_false_alarm_probability), float(failure_probability), bound)


def detect_ca_noncoherent_tolerance(iq, threshold):
    candidate = acquire_ca_noncoherent(iq, threshold.config)
    score = candidate.statistic if candidate is not None else 0.
    return CaNoncoherentDetection(candidate is not None and score > threshold.score, score, threshold.score, candidate)
