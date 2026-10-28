"""Detection calibration ranks, scale invariance and candidate preservation."""

from dataclasses import replace

import numpy as np
import pytest

from gnss_gpu.code_delay_cpu import CodeDelayEstimate, estimate_ca_delay
from gnss_gpu import code_detection_cpu as detection
from gnss_gpu.code_detection_cpu import (
    ca_detection_statistic, calibrate_ca_threshold, detect_ca_signal,
)
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.validation.tracking_residuals import generate_ca_code


def threshold():
    return calibrate_ca_threshold(np.arange(1., 100.), 2*CHIP_RATE,
                                   false_alarm_probability=.05)


def test_calibration_uses_finite_sample_rank():
    gate = calibrate_ca_threshold(np.arange(1., 100.), 2*CHIP_RATE,
                                  false_alarm_probability=.01)
    assert gate.order_statistic == 99
    assert gate.score == 99
    assert gate.marginal_false_alarm_bound == .01


def test_gate_rejects_equal_scores_even_with_nonzero_candidate(monkeypatch):
    gate = calibrate_ca_threshold(np.full(99, 1023.), 2*CHIP_RATE,
                                  false_alarm_probability=.05)
    candidate = CodeDelayEstimate(0., 2*CHIP_RATE, 0, 1+0j, .5)
    monkeypatch.setattr(detection, "estimate_ca_delay", lambda *args, **kwargs: candidate)
    assert not detect_ca_signal(np.ones(4092), gate).accepted
    candidate = replace(candidate, fit_residual_fraction=.49)
    assert detect_ca_signal(np.ones(4092), gate).accepted


def test_detection_score_is_iq_scale_invariant():
    rng = np.random.default_rng(12)
    iq = rng.normal(size=4092)
    a = estimate_ca_delay(iq*1e-6, 2*CHIP_RATE)
    b = estimate_ca_delay(iq*1e6, 2*CHIP_RATE)
    assert ca_detection_statistic(a) == pytest.approx(ca_detection_statistic(b), abs=1e-10)


def test_gate_retains_rejected_candidate_and_accepts_strong_signal():
    rng = np.random.default_rng(44)
    noise = rng.normal(size=4092)
    rejected = detect_ca_signal(noise, threshold())
    assert not rejected.accepted
    assert rejected.candidate is not None
    code = np.repeat(generate_ca_code(1), 2)
    iq = np.column_stack([code, np.zeros_like(code)]).ravel()
    accepted = detect_ca_signal(iq, threshold())
    assert accepted.accepted and accepted.candidate is not None
    assert accepted.statistic > accepted.threshold


def test_zero_observation_has_no_candidate_and_is_rejected():
    result = detect_ca_signal(np.zeros(4092), threshold())
    assert not result.accepted and result.candidate is None
    assert result.statistic == 0


@pytest.mark.parametrize("scores,alpha", [([], .01), ([0., 1.], .1),
                                            ([1., np.nan], .1), ([1., 2.], .01),
                                            ([1., 2.], 0.), ([1., 2.], 1.)])
def test_invalid_or_insufficient_calibration_is_rejected(scores, alpha):
    with pytest.raises(ValueError):
        calibrate_ca_threshold(scores, 2*CHIP_RATE, false_alarm_probability=alpha)


def test_invalid_search_configuration_and_corrupt_rank_metadata_are_rejected():
    gate = threshold()
    for kwargs in (dict(sampling_freq=2.6e6), dict(prn=33), dict(score=np.inf),
                   dict(order_statistic=1), dict(carrier_hz=2*CHIP_RATE)):
        with pytest.raises(ValueError):
            replace(gate, **kwargs)
    with pytest.raises(ValueError, match="1 ms"):
        detect_ca_signal(np.zeros(100), gate)
