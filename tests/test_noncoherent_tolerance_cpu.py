import math

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch, acquire_ca_noncoherent
from gnss_gpu.noncoherent_tolerance_cpu import (
    binomial_lower_tail, tolerance_order, calibrate_ca_noncoherent_tolerance, detect_ca_noncoherent_tolerance,
)


def test_log_domain_binomial_matches_direct_small_distribution():
    for upper in range(12):
        expected = sum(math.comb(20, j)*.2**j*.8**(20-j) for j in range(upper+1))
        assert binomial_lower_tail(20, .2, upper) == pytest.approx(expected, abs=2e-14)


@pytest.mark.parametrize("n,expected", [(1999, 1988), (4999, 4963)])
def test_tolerance_rank_and_its_minimality(n, expected):
    order, bound = tolerance_order(n)
    assert order == expected
    assert bound <= .025
    assert binomial_lower_tail(n, .01, n-order+1) > .025


@pytest.mark.parametrize("n,alpha,delta", [(99, .01, .025), (1000, 0., .025),
                                          (1000, .01, 1.), (1000, np.nan, .025)])
def test_insufficient_counts_and_invalid_probabilities(n, alpha, delta):
    with pytest.raises(ValueError):
        tolerance_order(n, alpha, delta)


def test_conservative_ties_and_bound_config_are_preserved():
    config = CaNoncoherentSearch(CaSearch(2046000., (0.,)), 1)
    iq = np.random.default_rng(5).normal(size=(1, 4092))
    score = acquire_ca_noncoherent(iq, config).statistic
    threshold = calibrate_ca_noncoherent_tolerance([score]*499, config)
    found = detect_ca_noncoherent_tolerance(iq, threshold)
    assert threshold.config == config
    assert not found.accepted and found.candidate is not None
    assert threshold.binomial_failure_bound <= .025
