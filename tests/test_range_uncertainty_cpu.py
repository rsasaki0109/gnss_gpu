"""Causal scatter estimation, smoothing correction and history boundaries."""

import numpy as np
import pytest

from gnss_gpu.range_uncertainty_cpu import CausalRangeUncertainty, RangeUncertaintyConfig


@pytest.mark.parametrize("gain", [.25, 1.])
def test_stationary_smoothed_noise_scatter_is_on_correct_scale(gain):
    cfg = RangeUncertaintyConfig(gain=gain, min_sigma_m=.01, initial_sigma_m=10.)
    model, rng, state, sigmas = CausalRangeUncertainty(cfg), np.random.default_rng(44219), 0., []
    input_sigma = 10.
    for index in range(4000):
        state = (1-gain)*state+gain*rng.normal(scale=input_sigma)
        result = model.update(1, 22e6+state, time_s=index*.001)
        if index > 100:
            sigmas.append(result.sigma_m)
    expected = input_sigma*np.sqrt(gain/(2-gain))
    assert .85 < np.mean(sigmas)/expected < 1.15


def test_current_outlier_cannot_inflate_its_own_prior_uncertainty():
    a, b = CausalRangeUncertainty(), CausalRangeUncertainty()
    for index in range(12):
        a.update(1, 22e6+index%3, time_s=index*.001)
        b.update(1, 22e6+index%3, time_s=index*.001)
    ordinary = a.update(1, 22e6, time_s=.012)
    outlier = b.update(1, 22e6+1e4, time_s=.012)
    assert ordinary == outlier and outlier.ready


def test_constant_range_bias_is_invisible_to_scatter_estimation():
    a, b = CausalRangeUncertainty(), CausalRangeUncertainty()
    for index in range(40):
        first = a.update(1, 22e6+index%3, time_s=index*.001)
        biased = b.update(1, 22e6+index%3+200., time_s=index*.001)
        assert first == biased
    assert first.ready


def test_gap_reset_and_prn_isolation():
    model = CausalRangeUncertainty()
    for i in range(12):
        result = model.update(1, 22e6+i%2, time_s=i*.001)
    assert result.ready
    cold = model.update(2, 22e6, time_s=.012)
    assert not cold.ready and cold.sigma_m == 10.
    restarted = model.update(1, 22e6, time_s=.03)
    assert restarted == cold
    model.reset()
    assert model.update(1, 22e6, time_s=0.) == cold


def test_invalid_input_does_not_advance_history():
    model = CausalRangeUncertainty()
    model.update(1, 22e6, time_s=0.)
    with pytest.raises(ValueError):
        model.update(1, np.nan, time_s=.001)
    assert model.update(1, 22e6, time_s=.001).prior_differences == 0
    with pytest.raises(ValueError):
        model.update(1, 22e6, time_s=.001)


@pytest.mark.parametrize("kwargs", [dict(gain=1.1), dict(min_differences=1), dict(window_differences=3),
                                      dict(min_sigma_m=11), dict(max_sigma_m=1), dict(max_gap_s=0)])
def test_invalid_model_settings(kwargs):
    with pytest.raises(ValueError):
        RangeUncertaintyConfig(**kwargs)
