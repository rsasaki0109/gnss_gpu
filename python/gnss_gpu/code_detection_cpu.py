"""Empirically calibrated single-PRN/code-period acquisition detection.

Calibration must use independent, representative noise-only observations scored
with the same sampling/PRN/carrier configuration and complete delay search.
The gate supplements estimate_ca_delay; it does not change its candidate output.
"""

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from gnss_gpu.code_delay_cpu import CodeDelayEstimate, estimate_ca_delay
from gnss_gpu.input_validation import positive_float, positive_int, finite_float
from gnss_gpu.signal_sim_cpu import CHIP_RATE, _sample_index


def ca_detection_statistic(candidate: Optional[CodeDelayEstimate]):
    """N * fitted coherent-energy fraction, invariant to overall IQ scaling.

    Calibration includes the strongest-peak selection and fractional fit. This
    is a detection statistic, not a C/N0 estimate or a ranging confidence score.
    """
    if candidate is None:
        return 0.
    return float(candidate.sampling_freq*.001*(1-candidate.fit_residual_fraction))


@dataclass(frozen=True)
class CaDetectionThreshold:
    sampling_freq: float
    prn: int
    carrier_hz: float
    score: float
    target_false_alarm_probability: float
    calibration_trials: int
    order_statistic: int

    def __post_init__(self):
        fs = positive_float("sampling_freq", self.sampling_freq)
        ratio = fs/CHIP_RATE
        if round(ratio) < 2 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-10):
            raise ValueError("sampling_freq must give >=2 integer samples/chip")
        prn = _sample_index("prn", self.prn)
        if not 1 <= prn <= 32:
            raise ValueError("prn must be in 1..32")
        carrier = finite_float("carrier_hz", self.carrier_hz)
        if abs(carrier) > fs/2:
            raise ValueError("carrier_hz must lie within the Nyquist interval")
        score = positive_float("score", self.score)
        probability = positive_float("target_false_alarm_probability", self.target_false_alarm_probability)
        if probability >= 1:
            raise ValueError("target_false_alarm_probability must be below one")
        trials = positive_int("calibration_trials", self.calibration_trials)
        order = positive_int("order_statistic", self.order_statistic)
        if order != math.ceil((trials+1)*(1-probability)) or order > trials:
            raise ValueError("calibration count/order cannot support this false-alarm target")
        for name, value in (("sampling_freq", fs), ("prn", prn), ("carrier_hz", carrier),
                            ("score", score), ("target_false_alarm_probability", probability),
                            ("calibration_trials", trials), ("order_statistic", order)):
            object.__setattr__(self, name, value)

    @property
    def marginal_false_alarm_bound(self):
        """Rank bound under exchangeable null scores, not a per-threshold guarantee."""
        return (self.calibration_trials+1-self.order_statistic)/(self.calibration_trials+1)


@dataclass(frozen=True)
class CaDetectionResult:
    accepted: bool
    statistic: float
    threshold: float
    candidate: Optional[CodeDelayEstimate]


def calibrate_ca_threshold(noise_statistics, sampling_freq, *, prn=1, carrier_hz=0.,
                           false_alarm_probability=.01):
    """Freeze an order-statistic threshold from independently scored nulls.

    For n calibration scores choose k=ceil((n+1)*(1-alpha)), the kth ascending
    value, and accept only scores strictly greater than it. With exchangeable
    calibration/evaluation nulls, the marginal false-alarm probability is at
    most (n+1-k)/(n+1); ties are rejected. This averages over random calibration
    sets, not a guarantee on a particular saved threshold or real RF noise.
    A target below the finite calibration resolution is rejected.
    """
    scores = np.asarray(list(noise_statistics), dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.all(np.isfinite(scores)) or np.any(scores <= 0):
        raise ValueError("noise_statistics must contain finite, positive noise-only scores")
    alpha = positive_float("false_alarm_probability", false_alarm_probability)
    if alpha >= 1:
        raise ValueError("false_alarm_probability must be below one")
    order = math.ceil((len(scores)+1)*(1-alpha))
    if order > len(scores):
        raise ValueError("more calibration trials are required for this false-alarm target")
    score = float(np.partition(scores, order-1)[order-1])
    return CaDetectionThreshold(sampling_freq, prn, carrier_hz, score, alpha, len(scores), order)


def detect_ca_signal(iq, threshold: CaDetectionThreshold):
    """Estimate a candidate and apply a frozen, matching search configuration.

    Rejected candidates remain available for diagnostics; callers must inspect
    accepted before using them. Acceptance does not validate the delay in
    multipath, fix code-period ambiguity, or cover extra PRNs/carrier searches.
    """
    candidate = estimate_ca_delay(iq, threshold.sampling_freq, prn=threshold.prn,
                                  carrier_hz=threshold.carrier_hz)
    statistic = ca_detection_statistic(candidate)
    return CaDetectionResult(candidate is not None and statistic > threshold.score,
                              statistic, threshold.score, candidate)
